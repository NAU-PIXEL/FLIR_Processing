
import os
from flirimageextractor import FlirImageExtractor
from osgeo import gdal, osr
from exiftool import ExifToolHelper

def process_flir_images(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all files in the input directory
    file_list = os.listdir(input_dir)

    # Process each file in the input directory
    for file_name in file_list:
        # Check if the file is a FLIR radiometric JPEG
        if file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg"):
            input_path = os.path.join(input_dir, file_name)

            # Process the FLIR image
            process_flir_image(input_path, output_dir)

def process_flir_image(input_path, output_dir):
    # Open the FLIR image using FlirImageExtractor
    flir_image = FlirImageExtractor()
    flir_image.loadfile(input_path)
    metadata = flir_image.get_metadata(input_path)

    # Get the visible and thermal images only when they exist
    visible_image = None
    thermal_image = None
    if "EmbeddedImageType" in metadata:
        visible_image = flir_image.extract_embedded_image()
    if "RawThermalImageType" in metadata:
        thermal_image = flir_image.extract_thermal_image()



    # Perform your image processing here (for thermal images)
    # For example, you can apply filters, resize, etc.

    # Get the input file name without extension
    file_name_without_extension = os.path.splitext(os.path.basename(input_path))[0]

    if thermal_image is not None:
        # Save the thermal image as a GeoTIFF with EXIF metadata
        save_geotiff(os.path.join(output_dir, file_name_without_extension + '_thermal.tif'), thermal_image, flir_image, metadata)

    # Check if the visible image is available before saving
    if visible_image is not None:
        # Perform your image processing here (for visible images)
        # For example, you can apply filters, resize, etc.

        # Save the visible image as a GeoTIFF with EXIF metadata
        save_geotiff(os.path.join(output_dir, file_name_without_extension + '_visible.tif'), visible_image, flir_image, metadata)

def save_geotiff(output_path, image_data, flir_image, metadata):
    # Create a GeoTIFF file
    driver: gdal.Driver = gdal.GetDriverByName('GTiff')

    # Thermal will have 1 band for temperature, visible will have 3 for RGB
    dataset: gdal.Dataset = driver.Create(output_path, image_data.shape[1], image_data.shape[0], image_data.ndim, gdal.GDT_Float32)

    # Set the spatial reference
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)  # WGS84
    dataset.SetProjection(spatial_ref.ExportToWkt())

    # Set the geotransform
    # dataset.SetGeoTransform((flir_image.longitude, flir_image.pixel_pitch, 0, flir_image.latitude, 0, -flir_image.pixel_pitch))

    # Write the image data to the GeoTIFF
    for band_n in range(image_data.ndim):
        if image_data.ndim != 2:
            band_data = image_data[:,:,band_n]
        else:
            band_data = image_data
        # bands count from 1
        band: gdal.Band = dataset.GetRasterBand(band_n + 1)
        band.WriteArray(band_data)

    # Set EXIF metadata
    for key, value in metadata.items():
        dataset.SetMetadataItem(key, str(value))

    # Close the dataset
    print(dataset)
    dataset.FlushCache()
    dataset = None

if __name__ == "__main__":
    # Replace 'input_directory' and 'output_directory' with your actual directories
    input_directory = 'input'
    output_directory = 'output'

    process_flir_images(input_directory, output_directory)

