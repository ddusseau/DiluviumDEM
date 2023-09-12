# DiluviumDEM

DiluviumDEM is a global coastal digital elevation model based on the CopernicusDEM (30m) but corrected up to 80 meters in elevation using a LightGBM model. The code to generate DiluviumDEM is published in this repository. DiluviumDEM can be downloaded at: https://doi.org/10.5281/zenodo.8329293 or as an Earth Engine asset here: https://code.earthengine.google.com/?asset=users/ddusseau/DiluviumDEM

The methodology used to create DiluviumDEM is described in the article *DiluviumDEM: Enhanced accuracy in global coastal digital elevation models* published in the journal Remote Sensing of Environment.

**train_data_earthengine_export_DiluviumDEMv2.py**: Script to export training data from Google Earth Engine.

**preprocess_data_DiluviumDEMv2.py**: Script to convert exported GeoTiff files into TFRecords.

**LightGBM_train_DiluviumDEMv2.py**: Script to train LightGBM model.

**model.txt**: LightGBM model used to predict CopernicusDEM error.

**predict_data_earthengine_export_DiluviumDEMv2.py**: Script to export global prediction data from Google Earth Engine.

**LightGBM_predict_DiluviumDEMv2.py**: Script to predict CopernicusDEM error using model.txt.
