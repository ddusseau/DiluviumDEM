import os
import ee
from google.cloud import storage


#preprocess data
def preprocess(region_id):
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
#    global ausgeoid09
#    ausgeoid09 = ee.Image('users/ddusseau/AUSGeoid09')

    noaa_lidar_egm08 = ee.ImageCollection("users/ddusseau/NOAA_Lidar_EGM08").mosaic()
    noaa_lidar_egm08 = noaa_lidar_egm08.updateMask(noaa_lidar_egm08.gte(-3))

    ahn3_wgs84 = ee.Image('users/ddusseau/AHN3_WGS84_EPSG4326')
    ned_egm08 = ahn3_wgs84.subtract(egm08_geoid)

    EW_lidar = ee.Image('users/ddusseau/England_Wales_10m_DTM')
    EW_lidar = EW_lidar.updateMask(EW_lidar.neq(0))
    ogm15_geoid = ee.Image('users/ddusseau/OSTN15_OSGM15_DataFile_Height')
    EW_lidar_egm08 = EW_lidar.add(ogm15_geoid).subtract(egm08_geoid)

    lidar_egm08 = ee.ImageCollection([noaa_lidar_egm08,ned_egm08,EW_lidar_egm08]).mosaic()

    ## Copdem ERROR
    copdem_error = copdem.subtract(lidar_egm08).select(['b1'],['error_val'])

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

    combined_all = ee.Image.cat([copdem_adjacent,copdem90_adjacent,canopy_simard,potapov_canopy,veg_cover_modis,veg_cover_nasa,pop_density,built_up,urban_cover,copdem_diff_05_1,copdem_diff_2_4,copdem_diff_1_2,copdem_sobel,sent_built_up,copdem_slope,copdem_aspect,EVI,cop_tree_cover,copdem_minus_merit,night_lights,l8,crop_cover,flooded_vegetation,VH,icesat2_filled,coperror_esa_filled,copdem_error])
    combined_all_masked = combined_all.float().reproject(projection).updateMask(final_mask)

    print(combined_all_masked.bandNames().getInfo())


#    Northeast_extent =  ee.Geometry.Polygon([[-69.9998611111111018,44.0001388888888911],[-72.0001388888888840,44.0001388888888911],[-72.0001388888888840,40.9998611111111160],[-69.9998611111111018,40.9998611111111160]])

#    combined_all_masked = combined_all_masked.clip(Northeast_extent)
#    export_region = quarter_deg_grid.filter(ee.Filter.eq("index_id", 24973)).first().geometry()
#    export_region = quarter_train_grids.filter(ee.Filter.eq("index_id", 1665)).geometry()
#    sampled_data = combined_all_masked.sampleRegions(collection=quarter_deg_grid,scale=scale)
#    print(data_sample.size().getInfo())

    test_SC = ee.Geometry.Polygon(
        [[[-80.9936882054576, 33.557365257408364],
          [-80.9936882054576, 32.270824426518644],
          [-79.0765739476451, 32.270824426518644],
          [-79.0765739476451, 33.557365257408364]]])

    test_MD = ee.Geometry.Polygon(
        [[[-80.9277702367076, 26.973429172254157],
          [-80.9277702367076, 25.525036213988386],
          [-79.5819450413951, 25.525036213988386],
          [-79.5819450413951, 26.973429172254157]]])

    test_NED = ee.Geometry.Polygon(
        [[[4.100916286729892, 52.69798732613559],
          [4.100916286729892, 51.708271339019696],
          [5.446741482042392, 51.708271339019696],
          [5.446741482042392, 52.69798732613559]]])


# Extract by the bounds of a quarter-degree grid to prevent memory errors
    USA_train_area = ee.Geometry.MultiPolygon(
    [[[[-126.1389869678728, 49.37943667272456],
       [-124.9085182178728, 46.37414265559945],
       [-125.2600807178728, 40.84364866369365],
       [-124.7327369678728, 38.13100911530016],
       [-122.3596900928728, 34.303418410741315],
       [-118.7561744678728, 31.34978547268597],
       [-115.2405494678728, 32.54301140271528],
       [-118.5803932178728, 35.52855597953824],
       [-120.7776588428728, 38.061841530375716],
       [-122.2717994678728, 40.10823939707025],
       [-122.2717994678728, 44.21047732697397],
       [-122.0081275928728, 47.8691186490533],
       [-120.6897682178728, 49.6077874901883]]],
     [[[-98.1677955616228, 27.36776218369218],
       [-99.3323463428728, 25.558196423756957],
       [-96.6077369678728, 24.44304389999903],
       [-96.6956275928728, 27.13335479410766],
       [-93.9710182178728, 29.071433473983472],
       [-91.1585182178728, 28.76370568684237],
       [-89.2249244678728, 28.53231236153669],
       [-87.5550025928728, 29.301628456392027],
       [-84.4788307178728, 29.14822251836185],
       [-83.5999244678728, 28.145526607451288],
       [-82.4573463428728, 25.320089883819808],
       [-85.0940650928728, 23.881713923290874],
       [-83.4241432178728, 22.670685903860754],
       [-80.0842994678728, 22.832789408674472],
       [-78.3264869678728, 25.081514258645708],
       [-80.6116432178728, 30.36898411856858],
       [-79.2053932178728, 32.46889011584799],
       [-74.8108619678728, 35.24193578134788],
       [-74.8108619678728, 37.71502273909855],
       [-73.0530494678728, 39.50056081585076],
       [-68.83429946787281, 41.37342429112619],
       [-69.44953384287281, 42.93703755974807],
       [-67.51594009287281, 44.147445119363574],
       [-65.67023696787281, 44.77473444018567],
       [-66.90070571787281, 46.191920192294035],
       [-72.4378150928728, 43.129769841418046],
       [-71.9544166553728, 42.16006039392763],
       [-75.7556861866228, 40.41006213458311],
       [-78.2385963428728, 39.568345743992175],
       [-78.3264869678728, 37.5758382353635],
       [-77.9749244678728, 35.885396224962186],
       [-82.3694557178728, 32.17179506879368],
       [-94.1028541553728, 32.52448680981781],
       [-95.5091041553728, 30.179223242156965],
       [-97.8162330616228, 28.74444242289556]]]])

    NED_train_area = ee.Geometry.Polygon(
    [[[3.0718859999064962, 51.51862768578977],
      [3.1487902967814962, 51.209951909046886],
      [3.8519152967814962, 51.09969879992004],
      [4.653917249906496, 51.1824133280692],
      [5.346055921781496, 51.04447342567036],
      [5.499864515531496, 50.60029519297746],
      [6.093126234281496, 50.60029519297746],
      [6.455675062406496, 50.83679098439246],
      [7.653184828031496, 53.222123392275876],
      [8.455186781156495, 53.84893512220299],
      [7.664171156156496, 53.984815255325856],
      [4.686876234281496, 53.43209205814361],
      [3.4344348280314962, 52.291318504196795]]])

    UK_train_area = ee.Geometry.Polygon(
    [[[-7.157435087782047, 50.26787687945536],
      [-7.552942900282047, 49.2458849496961],
      [-5.509485869032047, 49.13099908973419],
      [-3.048548369032047, 49.80213748992036],
      [1.258092255967953, 50.68736168312684],
      [1.873326630967953, 51.22713615751765],
      [2.115025849717953, 51.828601757175896],
      [2.290807099717953, 52.782452776549015],
      [1.038365693467953, 53.62472448655426],
      [-2.213587431532047, 56.2735141167583],
      [-4.894251494032047, 54.34818816290485],
      [-6.388392119032047, 51.13071506963743]]])


    AUS_train_area = ee.Geometry.Polygon(
        [[[116.30494555125124, -36.748083296680576],
          [121.75416430125124, -35.11132564944443],
          [128.43385180125122, -33.07354462587446],
          [132.74049242625122, -33.51433102004271],
          [134.93775805125122, -36.04061505033895],
          [136.69557055125122, -37.16946495900567],
          [138.80494555125122, -38.281700597448044],
          [141.79322680125122, -40.78923371087318],
          [143.63892992625122, -44.97291992469391],
          [149.79127367625122, -44.78608430552443],
          [151.54908617625122, -42.23701737905659],
          [150.14283617625122, -39.10487189378016],
          [153.57057055125122, -35.11132564944443],
          [155.85572680125122, -28.237084365694564],
          [154.09791430125122, -22.928496331163466],
          [147.76978930125122, -17.571191203986036],
          [144.86939867625122, -11.448207036573132],
          [142.14478930125122, -9.547070181274908],
          [139.24439867625122, -12.394141026135808],
          [136.08033617625122, -9.89358496082031],
          [130.80689867625122, -9.460385876103295],
          [126.23658617625124, -11.706514171404052],
          [120.69947680125124, -15.972366191818418],
          [114.89869555125124, -19.653399120965723],
          [111.29517992625124, -21.953693472006368],
          [111.03150805125124, -25.57262086582585],
          [111.82252367625124, -31.0627682885625],
          [113.05299242625124, -35.25499410114213],
          [114.72291430125124, -36.46586890159154]]])

    Mekong_train_area = ee.Geometry.Polygon(
    [[[104.06023770203464, 10.994793634426692],
      [104.06023770203464, 8.356570225913503],
      [107.03890591492518, 8.356570225913503],
      [107.03890591492518, 10.994793634426692]]])

    train_area = USA_train_area
#    deg_grids = ee.FeatureCollection('users/ddusseau/GEO1988-CopernicusDEM-RP-002_GridFile')
    half_deg_grids = ee.FeatureCollection('users/ddusseau/CopDEM_half_deg_grid_clipped_coastal_below60N')

    def inter_fun(feature):
        return feature.set('intersect',feature.intersects(train_area))

    ## list of export grid cells
    export_grids = [data_file['properties']['id'] for data_file in half_deg_grids.map(inter_fun).filter(ee.Filter.eq('intersect',True)).toList(count=1000000).getInfo()]

    export_grids.sort()
#    print(export_grids)


    deg_grids = ee.FeatureCollection('users/ddusseau/CopernicusDEM_1deg_grid')

    def inter_fun(feature):
        return feature.set('intersect',feature.intersects(train_area))

    ## list of export grid cells
    export_grids = [data_file['properties']['Id'] for data_file in deg_grids.map(inter_fun).filter(ee.Filter.eq('intersect',True)).toList(count=1000000).getInfo()]

    export_grids.sort()
    print(export_grids)
    print(pooop)


    total_points = 0

    done = [str(blob.name) for blob in storage_client.list_blobs('ddusseau-climate-risk', prefix='Improved_NASADEM/DiluviumDEMV2/'+region_id)]
    done = done[1:]

    already_done = []
    for i in done:
        if i[len(i)-3:] == 'tif':
            already_done.append(i.split('_')[3][:-4])

#    for g in range(1,(export_grids.size().getInfo()+1)):


#    todo = [72473]
    for g in export_grids:
        if g not in already_done: # and g in todo:
            print(g)
            export_region = half_deg_grids.filter(ee.Filter.eq("id", g)).geometry()
            image = combined_all_masked.clip(export_region)
            pixelscount = image.select('error_val').reduceRegion(reducer=ee.Reducer.count(),geometry=image.geometry(),scale=scale,bestEffort=True).get('error_val')
            print(g,pixelscount.getInfo())

            if (pixelscount.getInfo() != 0):
#                total_points = total_points + pixelscount.getInfo()
                export_task = ee.batch.Export.image.toCloudStorage(
                  image= image.unmask(-9999),
                  description=str(g),
                  bucket= 'ddusseau-climate-risk',
                  fileNamePrefix= 'Improved_NASADEM/train_data/DiluviumDEMV2/'+region_id+'/training'+'_'+str(g),
                  scale= scale,
                  maxPixels =10000000000,
                )
                export_task.start()

#            print(total_points)




    return


######
#### MAIN #########
#######

#ee.Authenticate()
ee.Initialize()

storage_client = storage.Client()

###preprocess data
#region_id = 'USA'
region_id = 'USAV2'
#region_id = 'USA_V2'
#region_id = 'USA_V3_icesat'
#region_id = 'AUS'

#region_id = 'test/CopDEM'
preprocess(region_id)
