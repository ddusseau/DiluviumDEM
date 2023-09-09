import sys, os, string, time, re, getopt, glob2, shutil, math
#import osr
import numpy as np
import pandas as pd
#from osgeo import gdal
#from osgeo import ogr
import datetime
import rasterio
from datetime import datetime, timedelta
from sklearn.utils import shuffle
import tensorflow as tf


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# from https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f
#__author__ = "Sangwoong Yoon"
def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.

    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.

    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.

    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))

    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank,
                               # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None

    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)

    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecord.gz'
    options = tf.io.TFRecordOptions(tf.io.TFRecordOptions(compression_type='GZIP'))
    writer = tf.io.TFRecordWriter(result_tf_file,options=options)
    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))

    # iterate over each sample,
    # and serialize it as ProtoBuf.

    input_feature_names = ['copdem1', 'copdem2', 'copdem3', 'copdem4', 'copdem5', 'copdem6', 'copdem7', 'copdem8', 'copdem9', 'copdem90_1', 'copdem90_2', 'copdem90_3',
        'copdem90_4', 'copdem90_5', 'copdem90_6', 'copdem90_7', 'copdem90_8', 'copdem90_9', 'canopy_simard', 'canopy_potapov', 'veg_cover_modis', 'veg_cover_nasa',
        'population', 'built_up', 'urban_cover', 'gauss_1', 'gauss_2', 'gauss_3', 'sobel', 'built', 'slope', 'aspect', 'evi', 'cop_tree_cover', 'm_minus_c', 'night_lights',
        'L8_B1', 'L8_B2', 'L8_B3', 'L8_B4', 'L8_B5', 'L8_B6', 'L8_B7', 'L8_B8', 'L8_B9', 'L8_B10', 'L8_B11', 'crop_cover', 'flooded_vegetation', 'palsar', 'icesat2', 'esa_error']

    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]

        d_feature = {}
        for feat in range(len(input_feature_names)):
            d_feature[input_feature_names[feat]] = _float_feature(x[feat])

        if Y is not None:
            d_feature['error_val'] = dtype_feature_y(y)

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print("Writing {} done!".format(result_tf_file))



#create array where each row is a pixel and each column is a variable. Check each input has a valid value
def preprocess(file_list,shard_size):
    for file in file_list:
        with rasterio.open(file) as file_src:
          data = file_src.read(list(range(1,file_src.count+1)))
          all_data = data.reshape((data.shape[0],-1)) ##reshape to 2d array

          all_data_9999 = np.any(all_data == -9999,axis=0) ##boolean NaN array
          all_data_filt_9999 = all_data[:,~all_data_9999] ##extract all non9999 pixels

          all_data_nan = np.any(np.isnan(all_data_filt_9999),axis=0)
          all_data_filt = all_data_filt_9999[:,~all_data_nan]

          all_data_filt = shuffle(all_data_filt.T)

          if all_data_filt.shape[0] == 0:
              continue


          num_vars = all_data_filt.shape[1]
          print(np.array([all_data_filt[:,-1]]).T.shape)


          if int(all_data_filt.shape[0] / shard_size) > 0:
              all_data_filt = np.array_split(all_data_filt,int(all_data_filt.shape[0] / shard_size)) #use len of array to calculate shard size
              for i in range(len(all_data_filt)):
                file_name = file[:-4]+'_'+str(i)
                np_to_tfrecords(all_data_filt[i][:,0:(num_vars-1)],np.array([all_data_filt[i][:,-1]]).T,file_name,verbose=True)
          else:
            if all_data_filt.shape[0] == None:
                  continue
            np_to_tfrecords(all_data_filt[:,0:(num_vars-1)],np.array([all_data_filt[:,-1]]).T,file[:-4]+'_0',verbose=True)



#          np_to_tfrecords(all_data_filt[:,0:23],np.array([all_data_filt[:,23]]).T,file_name,verbose=True)


#            np.savetxt(file_name+'.gz',all_data_filt,delimiter=',',fmt='%1.8f')

    return


#####
#MAIN
#####

### for gcloud runs
#tfrecord_files = glob2.glob('./train_data/V5/NED/*.tfrecord.gz')
#all_files = glob2.glob('./train_data/V5/NED/*.tif')
#completed_files = ['./train_data/V5/NED/training_'+tfrecord_file.split('_')[2]+'.tif' for tfrecord_file in tfrecord_files]
#remaining_files = list(filter(lambda i: i not in completed_files, all_files))
#remaining_files = remaining_files[::-1]
#print(remaining_files)


remaining_files = glob2.glob('/Volumes/My Passport for Mac/train_data_v2/USA/training_*.tif')
shard_size = 10000
preprocess(remaining_files,shard_size)
