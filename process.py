# Python builtin libraries
import argparse
import datetime
import os
import pathlib
import re
from multiprocessing import Pool
from functools import partial

# External libraries
import numpy as np
import tqdm
from flirimageextractor import FlirImageExtractor
from osgeo import gdal, osr

# Local modules
import dji_utils
import file_utils

def setup_dirs(input_dir, output_dir, output_to_non_empty = False):
    try: 
        if not os.path.exists(input_dir):
            raise NotADirectoryError

        # Create output directory if it doesn't exist1
        if os.path.exists(output_dir):
            if len(os.listdir(output_dir)) != 0 and not output_to_non_empty:
                input('Output directory is not empty. Press Enter to continue or Ctrl+C to exit.')
        else:
            os.makedirs(output_dir)
    except PermissionError:
        raise
    except NotADirectoryError:
        raise

def gps_str_to_dd(gps_str):
    # Regex to capture group of digits, then non digits, repeating until we get to the NSEW at end of GPS string
    gps_re = re.compile('(?P<deg>\d*)\D*(?P<min>\d*)\D*(?P<sec>\d*.\d*)\D*(?P<dir>[NSEW])')
    res = gps_re.search(gps_str).groupdict()
    dec_gps = float(res['deg']) + float(res['min']) / 60 + float(res['sec']) / (60 ** 2)
    if res['dir'] in "WS":
        dec_gps *= -1
    return dec_gps

def get_height(height_source, metadata):
    # we assume that we'll always be able to calculate a height from the the 
    # metadata we pass in, this might need to be changed if that isn't true.

    # note that the height_source is effectively a tagged union, but we have not
    # yet enforced this with a typing solution.
    if height_source['type'] == 'height':
        return height_source['val']
    elif height_source['type'] == 'elevation':
        if type(metadata['GPSAltitude']) == str:
            gps_alt = float(re.search('(\d*.?\d*)',metadata['GPSAltitude']).group(1))
        elif type(metadata['GPSAltitude']) == np.float64:
            gps_alt = metadata['GPSAltitude']
        ground_elevation = height_source['val']
        return gps_alt - ground_elevation
    elif height_source['type'] == 'elevation_map':
        if type(metadata['GPSLatitude']) == str:
            gps_lat = gps_str_to_dd(metadata['GPSLatitude'])
            gps_long = gps_str_to_dd(metadata['GPSLongitude'])
            gps_alt = float(re.search('(\d*.?\d*)',metadata['GPSAltitude']).group(1))
        elif type(metadata['GPSLatitude']) == np.float64:
            gps_lat = metadata['GPSLatitude']
            gps_long = metadata['GPSLongitude']
            gps_alt = metadata['GPSAltitude']
        ground_elevation = file_utils.get_heightmap_elevation(height_source['val'], gps_lat, gps_long)
        return gps_alt - ground_elevation

def gps_from_source(metadata, gps_source):
    # get DateTimeOriginal from metadata and convert using format="%Y:%m:%d %H:%M:%S"
    # The Pro camera puts a decimal second and time zone on the DateTimeOriginal
    # string, it is always 10 characters long. Looks like "".324-07:00".
    # strptime doesn't have any nice way to tell it to ignore it, so we slice it off.
    timestamp = metadata['DateTimeOriginal'][:-10]
    timestamp = datetime.datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")
    
    # GPS location data is not available every second, look up the closest matching time
    gps = gps_source.loc[(gps_source['datetime'] - timestamp).abs().idxmin()]
    metadata['GPSLatitude'] = gps['latitude']
    metadata['GPSLongitude'] = gps['longitude']
    metadata['GPSAltitude'] = gps['altitude']

def process_raw_image(input_dir, output_dir, height_source, gps_source, filename):
    input_path = os.path.join(input_dir, filename)
    # Open the FLIR image using FlirImageExtractor
    flir_image = FlirImageExtractor()
    flir_image.loadfile(input_path)
    flir_image.fix_endian = False
    metadata = flir_image.get_metadata(input_path)

    if gps_source is not None:
        gps_from_source(metadata, gps_source)

    # GPS may not always give heights that are above ground reference altitude
    # but obviously distance is always positive and is required for calculations
    # so default to 1 meter
    flir_image.default_distance = max(1, get_height(height_source, metadata))
    metadata['ObjectDistance'] = flir_image.default_distance

    # Get the visible and thermal images only when they exist
    visible_image = None
    thermal_image = None
    if "EmbeddedImageType" in metadata:
        visible_image = flir_image.extract_embedded_image()
    if "RawThermalImageType" in metadata:
        thermal_image = flir_image.extract_thermal_image()
    else:
        print("Thermal image not found, are you sure this is looking at thermal images?")

    # Get the input file name without extension
    file_name_without_extension = os.path.splitext(filename)[0]

    if thermal_image is not None:
        # Save the thermal image as a GeoTIFF with EXIF metadata
        file_utils.save_geotiff(os.path.join(output_dir, file_name_without_extension + '_thermal.tif'), thermal_image, 'GTiff', metadata)

    # Check if the visible image is available before saving
    if visible_image is not None:
        # Perform your image processing here (for visible images)
        # For example, you can apply filters, resize, etc.

        # Save the visible image as a GeoTIFF with EXIF metadata
        file_utils.save_geotiff(os.path.join(output_dir, file_name_without_extension + '_visible.jpg'), visible_image, 'JPEG', metadata)


def load_gps_table(gps_source_path, source_type):
    if source_type == "dji":
        return dji_utils.gps_table_from_images(gps_source_path)
    else:
        raise NotImplementedError

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=pathlib.Path, required=True,
                        help="Input directory of files to process.")
    parser.add_argument('-o', '--output', type=pathlib.Path, required=True,
                        help="Output diredctory of files to process. Will prompt user to confirm if output folder is not empty. Will create directory and parent directories if they do not exist.")
    parser.add_argument('-y', '--yes', action='store_true', 
                        help="Automatic yes to prompts, including clobbering output directory.")
    parser.add_argument('--n_thread', type=int, 
                        help='Number of threads to use in multiprocessing pool.')
    parser.add_argument('--dji_gps_source', type=pathlib.Path,
                        help='Instead of using GPS data extracted from the image to get latitude and longitude, use GPS data extracted from a series of DJI camera images')


    elevation_group = parser.add_mutually_exclusive_group(required=True)
    elevation_group.add_argument('--height', type=int, 
                        help='Average distance between subject and drone during flight, in integer meters. For more accurate positioning over non-level terrain when GPS is available, use --heightmap option. When terrain is fairly level and GPS is available, the --elevation option can also be used.')
    elevation_group.add_argument('--heightmap', type=pathlib.Path, 
                        help='Path to input raster elevation map. Used as reference ground height to calculate object distance using GPS position. The coordinates of the input photography must be entirely within the elevation map. This must be a single layer GeoTiff.')
    elevation_group.add_argument('--elevation', type=int, 
                        help='Elevation of ground at measurement site in integer meters. Uses GPS elevation but assumes that the ground is level. If significant elevation changes in the terrain are present the --heightmap option should be used instead.')


    # Used internally for arbitrary things to help with debugging transparency
    # We do not show it in the help to avoid having to explain it.
    # Using this argument in production is entirely unsupported.
    parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    process_args = {}
    if args.height:
        height_source = {'val': args.height, 'type': 'height'}
    elif args.heightmap:
        height_source = {'val': args.heightmap, 'type': 'elevation_map'}
    elif args.elevation:
        height_source = {'val': args.elevation, 'type': 'elevation'}
    input_directory = args.input
    output_directory = args.output

    gps_source = None
    if args.dji_gps_source:
        gps_table = load_gps_table(args.dji_gps_source, source_type='dji')
        gps_source = gps_table

    setup_dirs(input_directory, output_directory, args.yes)

    filelist = os.listdir(input_directory)

    process_partial = partial(process_raw_image, input_directory, output_directory, height_source, gps_source)
    if args.debug:
        for file in filelist:
            process_partial(file)
    else:
        with Pool(args.n_thread) as p:
            list(tqdm.tqdm(p.imap(process_partial, filelist)))