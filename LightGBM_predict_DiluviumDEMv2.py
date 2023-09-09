from numba import njit
import glob2
from multiprocessing import Pool
from datetime import datetime
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
import rasterio
import pandas as pd
import os
from google.cloud import storage
import ee
import lightgbm as lgb

@njit
def predictions_index(final_array_len, predictions, final_out, nodata_9999_array):
    count = 0
    for i in range(final_array_len):
        if nodata_9999_array[i] == False:
            final_out[i] = predictions[count]
#            if predictions[count] < -50:
#                print(predictions[count])
            count = count + 1

    return final_out


def predict_fun(file):
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time start =", dt_string, file)

    with rasterio.open(file) as predict_src:

        all_data = predict_src.read()

        col_names = ['copdem1', 'copdem2', 'copdem3', 'copdem4', 'copdem5', 'copdem6', 'copdem7', 'copdem8', 'copdem9', 'copdem90_1', 'copdem90_2', 'copdem90_3',
            'copdem90_4', 'copdem90_5', 'copdem90_6', 'copdem90_7', 'copdem90_8', 'copdem90_9', 'canopy_simard', 'canopy_potapov', 'veg_cover_modis',
            'veg_cover_nasa', 'population', 'built_up', 'urban_cover', 'gauss_1', 'gauss_2', 'gauss_3', 'sobel', 'built', 'slope', 'aspect', 'evi',
            'cop_tree_cover', 'm_minus_c', 'night_lights', 'L8_B1', 'L8_B2', 'L8_B3', 'L8_B4', 'L8_B5', 'L8_B6', 'L8_B7', 'L8_B8', 'L8_B9', 'L8_B10',
            'L8_B11', 'crop_cover', 'flooded_vegetation', 'palsar', 'icesat2', 'esa_error']

        out_x = all_data.shape[1]
        out_y = all_data.shape[2]
        all_data = all_data.reshape((all_data.shape[0],-1)) ##reshape to 2d array
        all_data_pd = pd.DataFrame(all_data.T,columns=col_names)

        new_cols = sorted(['copdem1', 'copdem2', 'copdem3', 'copdem4', 'copdem5', 'copdem6', 'copdem7', 'copdem8', 'copdem9', 'copdem90_1', 'copdem90_2',
            'copdem90_3', 'copdem90_4', 'copdem90_5', 'copdem90_6', 'copdem90_7', 'copdem90_8', 'copdem90_9', 'canopy_simard', 'canopy_potapov', 'veg_cover_modis',
            'veg_cover_nasa', 'population', 'built_up', 'urban_cover', 'gauss_2', 'gauss_3', 'sobel', 'built', 'slope', 'aspect', 'evi', 'cop_tree_cover', 'm_minus_c',
            'night_lights', 'L8_B1', 'L8_B2', 'L8_B3', 'L8_B4', 'L8_B5', 'L8_B6', 'L8_B7', 'L8_B8', 'L8_B9', 'L8_B10', 'L8_B11', 'crop_cover', 'flooded_vegetation', 'palsar', 'icesat2', 'esa_error'])
        all_data = all_data_pd[new_cols].to_numpy()
        nodata_9999_array = np.any(np.equal(all_data, -9999),axis=1)
#        nodata_nan_array = np.any(np.isnan(all_data),axis=0)

        all_data = all_data[~nodata_9999_array,:]
        print(all_data.shape)
        if all_data.shape[0] <= 1:
            return

        bst = lgb.Booster(model_file='model.txt')

        predictions = bst.predict(all_data)
#        print(predictions)
#        predictions = model.predict(all_data, batch_size = batch_size, verbose=0)

        final_array_len = nodata_9999_array.shape[0]
        final_out = np.full(final_array_len,-9999,dtype=np.float32)

        if np.ndim(np.squeeze(predictions)) == 0:
            predictions = predictions[0]
        else:
            predictions = np.squeeze(predictions)
        final_out = predictions_index(final_array_len, predictions, final_out, nodata_9999_array)

        final_out_2d = np.reshape(final_out, (out_x, out_y))


        profile_out = predict_src.profile
        profile_out.update(count=1)
        profile_out.update(nodata=-9999)
        with rasterio.open( './predictions_DiluviumV2/predicted_'+file.split('_')[4]+'.tif','w',**profile_out) as dst:
            dst.write(final_out_2d,1)
#        with rasterio.open('./prediction_dataV5/predicted_'f.split('_')[2],'w',**profile_out) as dst:
            dst.write(final_out_2d,1)

    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time finish =", dt_string, file)

    return


#########
### MODEL

if __name__ == "__main__":


    all_files_glob = glob2.glob('./prediction_data_DiluviumV2/*.tif')
    completed_files = glob2.glob('./predictions_DiluviumV2/*.tif')
#    all_files_glob = all_files_glob[::-1]

    completed_files = [p.split('_')[2][:-4] for p in completed_files]


    for f in all_files_glob:
        id = f.split('_')[4]
        if id in completed_files:
            continue

        predict_fun(f)
