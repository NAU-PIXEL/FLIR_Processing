import numpy as np
from osgeo import gdal, osr

def get_heightmap_elevation(heightmap, gps_lat, gps_long):
    # this seems bad since it's in each thread, but Gdal is smart enough that
    # it only actually opens the file once and shares it between them in the background
    dataset: gdal.Dataset = gdal.Open(str(heightmap))
    # we could always assume that the user is giving us our heightmap in
    # WGS84/NAD83 format but this makes it flexible
    proj = dataset.GetProjection() 
    transform = dataset.GetGeoTransform()
    # we can skip reprojecting the coordinate system if the source is already
    # in an appropriate projection
    if 'WGS84' not in proj and 'NAD83' not in proj:
        source_srs = osr.SpatialReference()
        source_srs.ImportFromWkt(osr.GetUserInputAsWKT("urn:ogc:def:crs:OGC:1.3:CRS84"))

        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(proj)

        transform = osr.CoordinateTransformation(source_srs, target_srs)

        mapx, mapy, *_ = transform.TransformPoint(gps_lat, gps_long)
    else:
        mapx, mapy = gps_long, gps_lat

    tranform_inverse = gdal.InvGeoTransform(transform)
    px, py = gdal.ApplyGeoTransform(tranform_inverse, mapx, mapy)
    elevation =  dataset.ReadAsArray(px, py, 1,1)
    return elevation[0][0]

def save_geotiff(output_path, image_data, drivertype, metadata):
    # Create a GeoTIFF file
    driver: gdal.Driver = gdal.GetDriverByName(drivertype)

    # Thermal will have 1 band for temperature, visible will have 3 for RGB
    # normalize to 3 dimension array so that everything else is simpler.
    if image_data.ndim == 2:
        image_data = np.expand_dims(image_data, axis=2)
    if drivertype == 'JPEG':
        # JPEG doesn't have a direct create method, it has to be copied from another dataset.
        mem_driver = gdal.GetDriverByName( 'MEM' )
        dataset = mem_driver.Create( '', image_data.shape[1], image_data.shape[0], image_data.shape[2], gdal.GDT_Byte)

    if drivertype == 'GTiff':
        dataset: gdal.Dataset = driver.Create(output_path, image_data.shape[1], image_data.shape[0], image_data.shape[2], gdal.GDT_Float32)

    # Set the spatial reference
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)  # WGS84
    dataset.SetProjection(spatial_ref.ExportToWkt())

    # Set the geotransform
    # dataset.SetGeoTransform((flir_image.longitude, flir_image.pixel_pitch, 0, flir_image.latitude, 0, -flir_image.pixel_pitch))

    # Write the image data to the GeoTIFF
    for band_n in range(image_data.shape[2]):
        band_data = image_data[:,:,band_n]
        # bands count from 1
        band: gdal.Band = dataset.GetRasterBand(band_n + 1)
        band.WriteArray(band_data)

    # Set EXIF metadata
    for key, value in metadata.items():
        dataset.SetMetadataItem(key, str(value))

    if drivertype == 'JPEG':
        # replace dataset reference from memory to jpeg driver
        dataset = driver.CreateCopy(output_path, dataset, 0)
    # Close the dataset
    dataset.FlushCache()
    dataset = None