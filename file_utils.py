import subprocess

import numpy as np
import pandas as pd
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
    
    # make sure that all keys in metadata are strings, not all Python/OSgeo
    # versions will automatically stringify.
    for key, val in metadata.items():
        metadata[key] = str(val)

    # Set GDAL metadata, this is somewhat redundant with the exif metadata
    # but we did create a little more from the processing settings
    dataset.SetMetadata(metadata)

    # Write the image data to the GeoTIFF
    for band_n in range(image_data.shape[2]):
        band_data = image_data[:,:,band_n]
        # bands count from 1
        band: gdal.Band = dataset.GetRasterBand(band_n + 1)
        band.WriteArray(band_data)

    if drivertype == 'JPEG':
        # replace dataset reference from memory to jpeg driver
        dataset = driver.CreateCopy(output_path, dataset, 0)
    # Close the dataset
    dataset.FlushCache()
    dataset = None

    # now that file is written, copy exif data from original using exiftool
    subprocess.run(['exiftool', '-overwrite_original', '-TagsFromFile', f"{metadata['Directory']}/{metadata['File Name']}", output_path])


def load_ground_data(ground_log_path, col_indices):
    rename_cols = ['datetime', "AtmosphericTemperature", "RelativeHumidity"]
    # For now we assume that the first row is some title stuff we don't care about
    # and that the column titles are the next row after that.
    # and since skiprows skips before getting the header we do that after
    df = pd.read_excel(str(ground_log_path), header=1)
    df = df.drop(index=[0,1])

    if type(col_indices[0]) == int:
        df = df.iloc[:, col_indices]
    else:
        df = df[col_indices]
    # input files all look like they have standardized datetime strings so
    # we're going to assume we don't have to use or pass a format string
    # we do not use df.infer_objects() so that we do not have to change
    # the behavior of the function that processes these fields, and it 
    # expects strings that it parses itself.
    df.columns = rename_cols
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['AtmosphericTemperature'] = df['AtmosphericTemperature'].astype('string')
    df['RelativeHumidity'] = df['RelativeHumidity'].astype('string')

    return df

