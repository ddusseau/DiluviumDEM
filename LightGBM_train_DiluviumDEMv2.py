import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import normalize
from tensorflow.keras import backend as K_backend
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os, random, glob2
from datetime import datetime
from google.cloud import storage
import json
import rasterio
from numba import njit
import pandas as pd
import tensorflow_datasets as tfds
import lightgbm as lgb
import math


def bias_rmse_fun(calc_bias, calc_rmse):
    ##Calculate bias.
    if calc_bias == 1:
        train_dataset_bias = train_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).batch(1000000).map(parse_tfrecord_bias,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
        sums = 0
        count = 0
        for element in train_dataset_bias.as_numpy_iterator():
            sums = sums + np.sum(element)
            count = count + element.shape[0]

        print("Training Bias: ",sums/count)

        val_dataset_bias = val_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).batch(1000000).map(parse_tfrecord_bias,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
        sums = 0
        count = 0
        for element in val_dataset_bias.as_numpy_iterator():
            sums = sums + np.sum(element)
            count = count + element.shape[0]

        print("Validation Bias: ",sums/count)

        test_dataset_bias = test_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).batch(1000000).map(parse_tfrecord_bias,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
        sums = 0
        count = 0
        for element in test_dataset_bias.as_numpy_iterator():
            sums = sums + np.sum(element)
            count = count + element.shape[0]

        print("Testing Bias: ",sums/count)

    ## Calculate RMSE.
    if calc_rmse == 1:
        train_dataset_rmse = train_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).batch(1000000).map(parse_tfrecord_rmse,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
        sums = 0
        count = 0
        for element in train_dataset_rmse.as_numpy_iterator():
            sums = sums + np.sum(element)
            count = count + element.shape[0]

        print("Training RMSE: ",sqrt(sums/count))

        val_dataset_rmse = val_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).batch(1000000).map(parse_tfrecord_rmse,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
        sums = 0
        count = 0
        for element in val_dataset_rmse.as_numpy_iterator():
            sums = sums + np.sum(element)
            count = count + element.shape[0]

        print("Validation RMSE: ",sqrt(sums/count))

        test_dataset_rmse = test_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).batch(1000000).map(parse_tfrecord_rmse,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
        sums = 0
        count = 0
        for element in test_dataset_rmse.as_numpy_iterator():
            sums = sums + np.sum(element)
            count = count + element.shape[0]

        print("Testing RMSE: ",sqrt(sums/count))

def parse_tfrecord_bias(example_proto):
    parsed_features = tf.io.parse_example(example_proto, features_dict)
    error_dif = parsed_features.pop(output_var)
    return error_dif

def parse_tfrecord_rmse(example_proto):
    parsed_features = tf.io.parse_example(example_proto, features_dict)
    error_dif = tf.math.square(parsed_features.pop(output_var))
    return error_dif


def parse_tfrecord_norm(example_proto):
    """The parsing function.

    Read a serialized example into the structure defined by featuresDict.

    Args:
      example_proto: a serialized Example.

    Returns:
      A tuple of the predictors dictionary and the label, cast to an `float32`.
    """
    parsed_features = tf.io.parse_single_example(example_proto, features_dict)

    pred_val = parsed_features.pop(output_var)

    return (tf.squeeze(list(parsed_features.values())), pred_val)






#########
### MODEL

if __name__ == "__main__":

    #set random seed to get consistent results
    ran_num = 1234
    np.random.seed(ran_num)
    tf.compat.v1.set_random_seed(ran_num)

    input_names = ['copdem1', 'copdem2', 'copdem3', 'copdem4', 'copdem5', 'copdem6', 'copdem7', 'copdem8', 'copdem9', 'copdem90_1', 'copdem90_2', 'copdem90_3', 'copdem90_4',
        'copdem90_5', 'copdem90_6', 'copdem90_7', 'copdem90_8', 'copdem90_9', 'canopy_simard', 'canopy_potapov', 'veg_cover_modis', 'veg_cover_nasa', 'population', 'built_up',
        'urban_cover', 'gauss_2', 'gauss_3', 'sobel', 'built', 'slope', 'aspect', 'evi', 'cop_tree_cover', 'm_minus_c', 'night_lights', 'L8_B1', 'L8_B2', 'L8_B3', 'L8_B4',
        'L8_B5', 'L8_B6', 'L8_B7', 'L8_B8', 'L8_B9', 'L8_B10', 'L8_B11', 'crop_cover', 'flooded_vegetation', 'palsar','icesat2', 'esa_error'] # 'gauss_1'



    global output_var
    output_var = 'error_val'

    feature_names = input_names + [output_var]


    input_len = len(input_names)

    # List of fixed-length features, all of which are float32
    columns = [
      tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in feature_names
    ]
    # Dictionary with names as keys, features as values.
    features_dict = dict(zip(feature_names, columns))
    batch_size = 10000


#    filenames = glob2.glob('./train_data/USA/training_*.tfrecord.gz')
    filenames = glob2.glob('/Volumes/My Passport for Mac/train_data_v2/*/*.gz')

#    filenames = glob2.glob('./NASADEM_model_USA_NED_JAP_V4_train_data/USA/training_*.tfrecord.gz')

    filenames_shuffled_all = np.array(shuffle(filenames,random_state=ran_num))
#    filenames_shuffled_all = filenames_shuffled_all[:1000]
    interval = 4500
    loops = math.ceil(len(filenames_shuffled_all)/interval)
    filenames_shuffled = filenames_shuffled_all[:interval] # first round

#    filenames_shuffled = filenames_shuffled_all

    train_filenames, val_filenames = train_test_split(filenames_shuffled,test_size=0.3, random_state=ran_num) # train: 70%, val: 30%

    out_val = val_filenames

#    print("Number of training files: ",train_filenames.shape[0])
#    print("Number of validation files: ",val_filenames.shape[0])


    train_filenames_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
    val_filenames_dataset = tf.data.Dataset.from_tensor_slices(val_filenames)


    train_dataset = train_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).map(parse_tfrecord_norm,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
    #

#    print(iter(train_dataset).next())

    val_dataset = val_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).map(parse_tfrecord_norm,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)



    train_dataset_np = tfds.as_numpy(train_dataset)

    input_x = []
    input_y = []
    for ex in train_dataset_np:
        input_x.extend(ex[0])
        input_y.extend(ex[1])

    input_y = np.array(input_y).squeeze()
    input_x = np.array(input_x)
    data_input = lgb.Dataset(data=input_x,label=input_y,feature_name=input_names)

    eval_dataset_np = tfds.as_numpy(val_dataset)

    eval_x = []
    eval_y = []
    for ex in eval_dataset_np:
        eval_x.extend(ex[0])
        eval_y.extend(ex[1])

    eval_y = np.array(eval_y).squeeze()
    eval_x = np.array(eval_x)
    eval_input = lgb.Dataset(data=eval_x,label=eval_y,feature_name=input_names)

#    print(np.array(input_x).shape)
#    print(np.array(input_y).shape)

    def mean_error(preds, train_data):
        y_true = train_data.get_label()
        return ('mean_error', np.mean(preds - y_true), False)


    params = {'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['rmse','mae','mse'],
        'learning_rate': 0.8935583206145743,
        'num_leaves': 2080,
        'max_depth': 16,
        'min_data_in_leaf': 600,
        'lambda_l1': 45,
        'lambda_l2': 35,
        'min_gain_to_split': 10,
        'bagging_fraction': 0.9,
        'bagging_freq': 10,
        'feature_fraction': 1.0,
        'verbose': 0,
        'n_estimators': 500,
        'seed':ran_num,
        'early_stopping_rounds':20,
        'first_metric_only':True
    }


    gbm = lgb.train(params,
                    data_input,
                    valid_sets=eval_input,
                    feval=mean_error,
                    keep_training_booster=True)
    gbm.save_model('model.txt')
#    print(poop)

    ## next rounds
    for i in range(1,loops):
        start_index = interval*i
        if i == loops - 1:
            end_index = len(filenames_shuffled_all) - 1
        else:
            end_index = interval*i+interval
        filenames_shuffled = filenames_shuffled_all[start_index:end_index]

        train_filenames, val_filenames = train_test_split(filenames_shuffled,test_size=0.3, random_state=ran_num) # train: 70%, val: 30%

        out_val = np.append(out_val,val_filenames)

    #    print("Number of training files: ",train_filenames.shape[0])
    #    print("Number of validation files: ",val_filenames.shape[0])


        train_filenames_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
        val_filenames_dataset = tf.data.Dataset.from_tensor_slices(val_filenames)


        train_dataset = train_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).map(parse_tfrecord_norm,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) #.cache()
        #

    #    print(iter(train_dataset).next())

        val_dataset = val_filenames_dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False).map(parse_tfrecord_norm,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)



        train_dataset_np = tfds.as_numpy(train_dataset)

        input_x = []
        input_y = []
        for ex in train_dataset_np:
            input_x.extend(ex[0])
            input_y.extend(ex[1])

        input_y = np.array(input_y).squeeze()
        input_x = np.array(input_x)
        data_input = lgb.Dataset(data=input_x,label=input_y,feature_name=input_names)

        eval_dataset_np = tfds.as_numpy(val_dataset)

        eval_x = []
        eval_y = []
        for ex in eval_dataset_np:
            eval_x.extend(ex[0])
            eval_y.extend(ex[1])

        eval_y = np.array(eval_y).squeeze()
        eval_x = np.array(eval_x)
        eval_input = lgb.Dataset(data=eval_x,label=eval_y,feature_name=input_names)


        keep_training_booster_val = True
        if i == loops - 1:
            keep_training_booster_val = False
        print(keep_training_booster_val)

        gbm = lgb.train(params,
                        data_input,
                        valid_sets=eval_input,
                        init_model='model.txt',
                        feval = mean_error,
                        keep_training_booster=keep_training_booster_val)
        gbm.save_model('model.txt')



    out_val_pd = pd.DataFrame(out_val,columns=['files'])


    out_val_pd.to_csv('validation_files.csv',index=None)






#    print(input_x.shape)
#
#    reg = lgb.LGBMRegressor(silent=False,random_state=0,verbose=1,metric=['rmse','mae','mse'],is_provide_training_metric=True,num_iterations=500,force_col_wise=True)
#    reg.fit(input_x,input_y,eval_set = [(eval_x,eval_y)])
#    print(reg.best_score)

#    calc_bias = 0
#    calc_rmse = 0
#    bias_rmse_fun(calc_bias,calc_rmse)
