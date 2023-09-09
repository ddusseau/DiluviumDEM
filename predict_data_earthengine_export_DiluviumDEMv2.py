import tensorflow as tf
import os, random, glob2
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout, GaussianNoise
from tensorflow.keras import regularizers
from tensorflow.keras.utils import normalize
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
#from numba import jit
#from numba.typed import List
#from sklearn.metrics import mean_squared_error
from math import sqrt
import ee
from google.cloud import storage

storage_client = storage.Client()

#preprocess data
def preprocess():
    global scale
    water = ee.ImageCollection('users/ddusseau/CopernicusDEM_WBM').mosaic().gt(0)
#    projection = water.projection()
    projection = ee.ImageCollection("projects/sat-io/open-datasets/GLO-30").first().projection()
    scale = projection.nominalScale().getInfo()
    print(scale)
    copdem = ee.ImageCollection("projects/sat-io/open-datasets/GLO-30").mosaic()

    copdem = copdem.setDefaultProjection(crs=projection)

#    global egm96_geoid
    egm96_geoid = ee.Image('users/ddusseau/egm96_15')
    egm08_geoid = ee.Image('users/ddusseau/egm2008-1')
#    egm96_2008_und = egm96_geoid.subtract(egm2008_geoid)


    ##CopDEM and surrounding pixels, creates image with 9 bands
    copdem_adjacent = copdem.neighborhoodToBands(kernel=ee.Kernel.square(radius=1,units='pixels')).select(['b1_-1_-1', 'b1_-1_0', 'b1_-1_1', 'b1_0_-1', 'b1_0_0', 'b1_0_1', 'b1_1_-1', 'b1_1_0', 'b1_1_1'],['copdem1','copdem2','copdem3','copdem4','copdem5','copdem6','copdem7','copdem8','copdem9'])

    ##CopDEM 90m and surrounding pixels, creates image with 9 bands
    scale90 = scale * 3
    copdem90_adjacent = copdem.reduceResolution(reducer=ee.Reducer.mean()).reproject(crs=projection,scale=scale90).neighborhoodToBands(kernel=ee.Kernel.square(radius=1,units='pixels')).select(['b1_-1_-1', 'b1_-1_0', 'b1_-1_1', 'b1_0_-1', 'b1_0_0', 'b1_0_1', 'b1_1_-1', 'b1_1_0', 'b1_1_1'],['copdem90_1','copdem90_2','copdem90_3','copdem90_4','copdem90_5','copdem90_6','copdem90_7','copdem90_8','copdem90_9'])

    ##CANOPY HEIGHT (Simard et al., 2011)
    canopy_simard = ee.Image("NASA/JPL/global_forest_canopy_height_2005").select(['1'],['canopy_simard'])

    ##CANOPY HEIGHT (Potapov et al., 2020)
    potapov_gedi = ee.ImageCollection('users/potapovpeter/GEDI_V27').mosaic()
    boreal_bounds = ee.Geometry.BBox(-180,52,180,75)
    potapov_boreal = ee.ImageCollection('users/potapovpeter/GEDI_V25_Boreal').mosaic().clip(boreal_bounds)
    potapov_canopy = ee.ImageCollection([potapov_gedi, potapov_boreal]).mosaic().select(['b1'],['canopy_potapov'])

    ##VEGETATION COVER
    # data from MODIS (250m) get data from 2011 to 2014 which is when TanDEM captured data and then get the best quality image, unmask no data areas to 0% tree cover
    veg_cover_modis = ee.ImageCollection("MODIS/006/MOD44B").filterDate('2011', '2015').qualityMosaic('Quality').unmask(0).select(['Percent_Tree_Cover'],['veg_cover_modis'])


    # Rescaled MOD44B data using Landsat data (30m), get year 2015, unmask no data areas to 0% tree cover. Just as accurate as MOD44B product and better in some areas according to publication.
    veg_cover_nasa = ee.ImageCollection("NASA/MEASURES/GFCC/TC/v3").filterDate('2015').mosaic().select('tree_canopy_cover').unmask(0).rename('veg_cover_nasa')


    ##POPULATION DENSITY
    #get the 2011 - 2014 population density since this is when the SRTM data was taken
    pop_density = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2011','2014').mosaic().log().unmask(0)

    ##BUILT-UP AREA
    ## take the percent confidence that the area is built up
    built_up = ee.Image('JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1').select(['cnfd'],['built_up'])

    ## Urban Land Cover
    urban_cover = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global").select(['urban-coverfraction'],['urban_cover']).filterDate('2015').mosaic().unmask(0)

    ## Built Up Area
    sent_built_up = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate('2016','2017').mosaic().unmask(0).select('built')


    ## Gaussian filters for urban detection
    gauss_1 = ee.Kernel.gaussian(radius=3,sigma=1,normalize=False)
    gauss_05 = ee.Kernel.gaussian(radius=3,sigma=0.5,normalize=False)
    gauss_2 = ee.Kernel.gaussian(radius=3,sigma=2,normalize=False)
    gauss_4 = ee.Kernel.gaussian(radius=3,sigma=4,normalize=False)
    copdem_diff_05_1 = copdem.convolve(gauss_05).subtract(copdem.convolve(gauss_1)).rename('gauss_1')
    copdem_diff_2_4 = copdem.convolve(gauss_2).subtract(copdem.convolve(gauss_4)).rename('gauss_2')
    copdem_diff_1_2 = copdem.convolve(gauss_1).subtract(copdem.convolve(gauss_2)).rename('gauss_3')
    copdem_sobel = copdem.convolve(ee.Kernel.sobel(normalize=False)).rename('sobel')

    ## Nighttime Lights
    night_lights = ee.ImageCollection('users/ddusseau/NASA_BlackMarble').mosaic().select('b1').unmask(0).rename('night_lights')

    ##SLOPE, apply convolution to smooth out noise and better capture signal
    copdem_slope = ee.Terrain.slope(copdem) #.convolve(ee.Kernel.gaussian(radius=7))

    ##ASPECT
    copdem_aspect = ee.Terrain.aspect(copdem)

    ##EVI check if filtering the date helps
    s2 = ee.ImageCollection('COPERNICUS/S2_SR').filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 30.0).sort('CLOUDY_PIXEL_PERCENTAGE',False).select('B2','B4','B8','B11').mosaic()
    EVI = s2.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',{'NIR': s2.select('B8'),'RED': s2.select('B4'),'BLUE': s2.select('B2')}).select(['constant'],['evi'])

    ##Copernicus Land Cover Tree Cover
    cop_tree_cover = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global").select(['tree-coverfraction'],['cop_tree_cover']).filterDate('2015').mosaic().unmask(0)

    ## CROP LAND COVER, there is some correlation between crop cover and error
    crop_cover = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global").select(['crops-coverfraction'],['crop_cover']).filterDate('2015').mosaic().unmask(0)

    ## Flooded Vegatation
    flooded_vegetation = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate('2016','2017').mosaic().unmask(0).select('flooded_vegetation')

    ##Error of CopDEM compared to MERIT
    merit = ee.Image("MERIT/DEM/v1_0_3").add(egm96_geoid).subtract(egm08_geoid)
    copdem_minus_merit = copdem.subtract(merit).select(['b1'],['m_minus_c'])

    ##Landsat 8, all 11 bands
    l8 = ee.ImageCollection("LANDSAT/LC08/C01/T1").filterMetadata('CLOUD_COVER_LAND', 'less_than', 23).sort('CLOUD_COVER_LAND',False).sort('IMAGE_QUALITY').select(['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11'],['L8_B1','L8_B2','L8_B3','L8_B4','L8_B5','L8_B6','L8_B7','L8_B8','L8_B9','L8_B10','L8_B11']).mosaic()


    ## Sentinel-1 Backscatter
    VH = ee.ImageCollection("projects/sat-io/open-datasets/S1GBM/normalized_s1_backscatter_VH").mosaic().rename('palsar')


    ##ICESat2 DEM
    icesat2 = ee.Image('users/ddusseau/ICESat2_GEDI_1km_combined')
    copdem_minus_icesat2 = copdem.subtract(icesat2)
    proj_5km = ee.Image('users/ddusseau/ICESat2_GLL_DTM_v1_200626').projection()
    icesat2_reduced = copdem_minus_icesat2.reduceResolution(reducer= ee.Reducer.mean(),maxPixels= 65536).reproject(crs= proj_5km)
    icesat2_filled = ee.ImageCollection([copdem_minus_icesat2,icesat2_reduced.updateMask(copdem_minus_icesat2.mask().eq(0))]).mosaic().rename("icesat2")


    ## CopernicusDEM published error
    coperror_esa_proj = ee.ImageCollection('users/ddusseau/CopernicusDEM_HEM').first().projection()
    coperror_esa = ee.ImageCollection('users/ddusseau/CopernicusDEM_HEM').mosaic().select(['b1'],['coperror_esa'])
    coperror_esa = coperror_esa.updateMask(coperror_esa.neq(-32767)).setDefaultProjection(coperror_esa_proj)
    icesat2_proj = ee.Image('users/ddusseau/ICESat2_GEDI_1km_combined').projection()
    coperror_esa_reduced = coperror_esa.reduceResolution(reducer= ee.Reducer.mean(),maxPixels= 65536).reproject(crs= icesat2_proj)
    coperror_esa_filled = ee.ImageCollection([coperror_esa,coperror_esa_reduced.updateMask(coperror_esa.mask().eq(0))]).mosaic().rename("esa_error")


    ##creates mask where copdem is in between -10 and 150 meters
    final_mask = copdem.expression('im > -10 && im < 80 && water == 0', {'im':copdem,'water':water})

    combined_all = ee.Image.cat([copdem_adjacent,copdem90_adjacent,canopy_simard,potapov_canopy,veg_cover_modis,veg_cover_nasa,pop_density,built_up,urban_cover,copdem_diff_05_1,copdem_diff_2_4,copdem_diff_1_2,copdem_sobel,sent_built_up,copdem_slope,copdem_aspect,EVI,cop_tree_cover,copdem_minus_merit,night_lights,l8,crop_cover,flooded_vegetation,VH,icesat2_filled,coperror_esa_filled])
    combined_all_masked = combined_all.float().reproject(projection).updateMask(final_mask)

    print(combined_all_masked.bandNames().getInfo())

#    def count_map(feature):
#        count = ee.Number(combined_all_masked.select('nasadem1').reduceRegion(reducer=ee.Reducer.count(),geometry=feature.geometry(),scale=scale,maxPixels=900000000000,tileScale=4).get('nasadem1'))
#
#        return feature.set('count',count)
#
#    count_export_regions = deg_grids.map(count_map)
#    export_task = ee.batch.Export.table.toCloudStorage(
#      collection= count_export_regions,
#      bucket= 'ddusseau-climate-risk',
#      fileNamePrefix= 'Improved_NASADEM/prediction_data_V3/counts'
#    )
#    export_task.start()
#    print(poop)


### Image export data
    deg_grids = ee.FeatureCollection('users/ddusseau/CopernicusDEM_1deg_grid')
    ids = [int(data_file['properties']['Id']) for data_file in deg_grids.toList(count=1000000).getInfo()]
    ids.sort()

    redo = []

    already_done = [str(blob.name) for blob in storage_client.list_blobs('ddusseau-climate-risk', prefix='Improved_NASADEM/prediction_data_DiluviumV2/')]
    already_done = already_done[1:]
    already_done = [int(t.split('_')[5]) for t in already_done]

    todo = [g for g in ids if g not in already_done]

    for g in todo:
        print(g)
        export_region = deg_grids.filter(ee.Filter.eq("Id",g)).geometry()
        pixelscount = combined_all_masked.select('copdem1').reduceRegion(reducer=ee.Reducer.count(),geometry=export_region,scale=scale,maxPixels =10000000000000,tileScale=2).get('copdem1')
        if pixelscount.getInfo() > 0:
            print(g,pixelscount.getInfo())
            export_task = ee.batch.Export.image.toCloudStorage(
                image= combined_all_masked.unmask(-9999),
                fileNamePrefix= 'Improved_NASADEM/prediction_data_DiluviumV2/predict_data_'+str(g)+'_tfrecord',
                scale= scale,
                bucket='ddusseau-climate-risk',
                region= export_region,
                maxPixels =10000000000000,
                description=str(g)
            )
            export_task.start()




    return


######
#### MAIN #########
#######

#ee.Authenticate()
ee.Initialize()


preprocess()
