# DiluviumDEM

DiluviumDEM is a best-in-class global coastal digital elevation model. The code to generate DiluviumDEM is published in this repository. DiluviumDEM can be downloaded at: 10.5281/zenodo.8329294

train_data_earthengine_export_DiluviumDEMv2.py: Script to export training data from Google Earth Engine.

preprocess_data_DiluviumDEMv2.py: Script to convert exported GeoTiff files into TFRecords.

LightGBM_train_DiluviumDEMv2.py: Script to train LightGBM model.

model.txt: LightGBM model used to predict CopernicusDEM error.

predict_data_earthengine_export_DiluviumDEMv2.py: Script to export global prediction data from Google Earth Engine.

LightGBM_predict_DiluviumDEMv2.py: Script to predict CopernicusDEM error using model.txt.
