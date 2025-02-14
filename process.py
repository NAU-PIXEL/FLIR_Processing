# Python builtin libraries
import argparse
import datetime
import os
import pathlib
import re
import subprocess
from multiprocessing import Pool
from functools import partial

# External libraries
import numpy as np
import pandas as pd
import tqdm
from flirimageextractor import FlirImageExtractor
from thermal_parser import Thermal
from osgeo import gdal, osr

# Local modules
import dji_utils
import file_utils
import flir_image_extractor_patch
import flir_pipeline

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
    gps_re = re.compile(r'(?P<deg>\d*)\D*(?P<min>\d*)\D*(?P<sec>\d*.\d*)\D*(?P<dir>[NSEW])')
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
            gps_alt = float(re.search(r'(\d*.?\d*)',metadata['GPSAltitude']).group(1))
        elif type(metadata['GPSAltitude']) == np.float64:
            gps_alt = metadata['GPSAltitude']
        ground_elevation = height_source['val']
        return gps_alt - ground_elevation
    elif height_source['type'] == 'elevation_map':
        if "GPSLatitude" in metadata.keys():
            lat_key = "GPSLatitude"
            long_key = "GPSLongitude"
            alt_key = "GPSAltitude"
        if "GPS Latitude" in metadata.keys():
            lat_key = "GPS Latitude"
            long_key = "GPS Longitude"
            alt_key = "GPS Altitude"
        if type(metadata[lat_key]) == str:
            gps_lat = gps_str_to_dd(metadata[lat_key])
            gps_long = gps_str_to_dd(metadata[long_key])
            gps_alt = float(re.search(r'(\d*.?\d*)',metadata[alt_key]).group(1))
        elif type(metadata[lat_key]) == np.float64:
            gps_lat = metadata[lat_key]
            gps_long = metadata[long_key]
            gps_alt = metadata[alt_key]
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

def atmos_from_source(flir_obj, metadata, atmos_source):
    timestamp = metadata['DateTimeOriginal'][:-10]
    timestamp = datetime.datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")

    # will be a df if the source is timestamped data to match
    if type(atmos_source) == type(pd.DataFrame()):
        atmos = atmos_source.loc[(atmos_source['datetime'] - timestamp).abs().idxmin()]
        atmos_dict = atmos.to_dict()
    # will just be a dict if the source is static data to apply
    else:
        atmos_dict = atmos_source
    for key, val in atmos_dict.items():
        metadata[key] = val
    flir_image_extractor_patch.monkey_patch_flir(flir_obj, atmos_dict)


def process_raw_image(input_dir, output_dir, data_sources, filename):
    input_path = os.path.join(input_dir, filename)
    # Open the FLIR image using FlirImageExtractor
    flir_image = FlirImageExtractor()
    flir_image.loadfile(input_path)
    flir_image.fix_endian = False
    metadata = flir_image.get_metadata(input_path)

    if 'gps' in data_sources:
        gps_from_source(metadata, data_sources['gps'])

    # in works for both dicts and dataframes
    if "atmos" in data_sources:
        atmos_from_source(flir_image, metadata, data_sources['atmos'])
    # GPS may not always give heights that are above ground reference altitude
    # but obviously distance is always positive and is required for calculations
    # so default to 1 meter
    flir_image.default_distance = max(1, get_height(data_sources['height'], metadata))
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

    # Check if the visible image is available before saving
    if visible_image is not None:
        # Perform your image processing here (for visible images)
        # For example, you can apply filters, resize, etc.

        # Save the visible image as a GeoTIFF with EXIF metadata
        file_utils.save_geotiff(os.path.join(output_dir, file_name_without_extension + '_visible.jpg'), visible_image, 'JPEG', metadata)

    if thermal_image is not None:
        # Save the thermal image as a GeoTIFF with EXIF metadata
        return (thermal_image, metadata)
        # file_utils.save_geotiff(os.path.join(output_dir, file_name_without_extension + '_thermal.tif'), thermal_image, 'GTiff', metadata)

# DJI thermal images are only thermal images with the array of data we care about embedded in their exif data, so we use a different processing path
# in a different code base that wraps the DJI SDK.
def process_dirp_image(input_dir, output_dir, data_sources, filename):

    input_path = os.path.join(input_dir, filename)
    # Open the FLIR image using FlirImageExtractor
    thermal = Thermal(dtype=np.float32)

    meta = subprocess.Popen([thermal._filepath_exiftool, input_path], stdout=subprocess.PIPE).communicate()[0]
    meta = meta.decode('utf8').replace('\r', '')
    metadata = dict([
        (field.split(':')[0].strip(), field.split(':')[1].strip()) for field in meta.split('\n') if ':' in field
    ])
    meta_kwargs = dict()
    # is this a reasonable default emissivity?
    meta_kwargs['emissivity'] = 0.98
    meta_kwargs['reflected_apparent_temperature'] = 20
    meta_kwargs['relative_humidity'] = 20

    if 'gps' in data_sources:
        gps_from_source(metadata, data_sources['gps'])

    # in works for both dicts and dataframes
    if "atmos" in data_sources:
        atmos_from_source(thermal, metadata, data_sources['atmos'])
        meta_kwargs["relative_humidity"] = metadata["RelativeHumidity"]
    # GPS may not always give heights that are above ground reference altitude
    # but obviously distance is always positive and is required for calculations
    # so default to 1 meter
    metadata['ObjectDistance'] = max(1, get_height(data_sources['height'], metadata))
    meta_kwargs['object_distance'] = metadata['ObjectDistance']

    thermal_image = thermal.parse_dirp2(input_path, **meta_kwargs)

    if thermal_image is not None:
        metadata['FileName'] = metadata['File Name']
        # Save the thermal image as a GeoTIFF with EXIF metadata
        return (thermal_image, metadata)
        # file_utils.save_geotiff(os.path.join(output_dir, file_name_without_extension + '_thermal.tif'), thermal_image, 'GTiff', metadata)


def load_gps_table(gps_source_path, source_type, offset):
    if source_type == "dji":
        return dji_utils.gps_table_from_images(gps_source_path, offset)
    else:
        raise NotImplementedError

if __name__ == "__main__":

    # It would probably make sense to break this into at least 2 submodules,
    # one that can do processing from the initial embedded jpegs and one that 
    # starts from the C converted images to apply post processing.
    pre = argparse.ArgumentParser(add_help=False)

    # This is kind of just a dummy arg that lets us override later behaviors
    # the actual argument is also in the main parser.
    # I don't like this as a long term solution, but it's okay for now.
    pre.add_argument('--post_process_help', action='store_true')
    pre_args, _ = pre.parse_known_args()
    required_override = False
    if pre_args.post_process_help:
        required_override = True

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=pathlib.Path, required=not required_override,
                        help="Input directory of files to process.")
    parser.add_argument('-o', '--output', type=pathlib.Path, required=not required_override,
                        help="Output diredctory of files to process. Will prompt user to confirm if output folder is not empty. Will create directory and parent directories if they do not exist.")
    parser.add_argument('-e', '--emissivity', type=float,
                        help="Emissivity used for calculation of temperature.")
    parser.add_argument('-y', '--yes', action='store_true', 
                        help="Automatic yes to prompts, including clobbering output directory.")
    parser.add_argument('--n_thread', type=int, 
                        help='Number of threads to use in multiprocessing pool.')
    parser.add_argument('--dji_gps_source', type=pathlib.Path,
                        help='Instead of using GPS data extracted from the image to get latitude and longitude, use GPS data extracted from a series of DJI camera images')
    parser.add_argument('--dji_time_offset', type=int, default=0,
                        help='Time offset in seconds to add to DJI timestamps. Useful if DJI timestamps are not synced to the FLIR camera\'s timestamp. Must be an integer.')
    parser.add_argument('--ground_station_log', type=pathlib.Path,
                        help='Excel file of ground station atmospheric conditions, for more accurate temperature calculations.')
    parser.add_argument('--ground_station_cols', type=str,
                        help='Comma separated list of either column titles or column indices. Required if ground_station_log is passed. The order must be timestamp,atmos_temp,relative_humidity. Integer indices count from 0, and string indices are case sensitive.')
    
    parser.add_argument('--dirp', action='store_true', 
                        help='Process DJI thermal images, known as "DJI IR processing" in the DJI SDK.')

    parser.add_argument('--post_process', action='store_true',
                        help='Perform destriping and then flat field correction on thermal images before output.')
    parser.add_argument('--post_pipeline', type=str,
                         help='Comma separted list of post process functions to apply, in order of application. Run with --post_process_help for more information.')
    parser.add_argument('--post_process_help', action='store_true',
                        help="Show information on loaded post processing functions and the names to use for them in the pipeline specification")

    elevation_group = parser.add_mutually_exclusive_group(required=not required_override)
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

    if args.post_process_help:
        flir_pipeline.pipeline.pipeline_help()
        exit(0)

    input_directory = args.input
    output_directory = args.output

    data_sources = dict()
    if args.height:
        height_source = {'val': args.height, 'type': 'height'}
    elif args.heightmap:
        height_source = {'val': args.heightmap, 'type': 'elevation_map'}
    elif args.elevation:
        height_source = {'val': args.elevation, 'type': 'elevation'}
    data_sources['height'] = height_source

    gps_source = None
    if args.dji_gps_source:
        gps_table = load_gps_table(args.dji_gps_source, 'dji', args.dji_time_offset)
        data_sources['gps'] = gps_table

    atmos_override = False
    atmos_source = dict()
    if args.ground_station_log:
        atmos_override = True
        if not args.ground_station_cols:
            raise ValueError('Told to use ground station logger without specifying columns, exiting early.')
        gs_cols = args.ground_station_cols.split(',')
        try:
            gs_cols = list(map(int, gs_cols))
        except ValueError:
            print('Ground station columns were not indices, using them as column names')
        atmos_source = file_utils.load_ground_data(args.ground_station_log, gs_cols)
    
    if args.emissivity:
        atmos_override = True
        # if atmos_source is still a dict, set the key
        # but if it's a dataframe from the ground station, this makes a column
        atmos_source['Emissivity'] = args.emissivity
        
    if atmos_override:
        data_sources['atmos'] = atmos_source

    if args.post_pipeline or args.post_process:
        if args.post_process:
            pipe_fns = ['destripe', 'ffc_s']
        elif args.post_pipeline:
            pipe_fns = args.post_pipeline.split(',')
        verify = flir_pipeline.pipeline.verify_pipeline_fns(pipe_fns)
        if not verify:
            raise LookupError('Functions in pipeline were not available, please double check --post_process_help to see which functions are available.')
        fn_pipeline = flir_pipeline.pipeline.build_pipeline(pipe_fns)

    setup_dirs(input_directory, output_directory, args.yes)

    filelist = os.listdir(input_directory)

    # filter filelist to remove non-jpegs
    filelist = [f for f in filelist if f.lower().endswith('.jpg') or f.endswith('.jpeg')]

    if args.dirp:
        process_function = process_dirp_image
    else:
        process_function = process_raw_image
    process_partial = partial(process_function, input_directory, output_directory, data_sources)
    if args.debug:
        res = []
        for file in filelist:
            res.append(process_partial(file))
    else:
        with Pool(args.n_thread) as p:
            res = list(tqdm.tqdm(p.imap(process_partial, filelist)))

    thermal, meta = list(zip(*res))
    thermal_ndarr = np.stack(thermal, axis=2)
    meta_df = pd.DataFrame(meta)
    
    if args.post_process or args.post_pipeline:
        
        print("Post Processing")
        thermal_ndarr = fn_pipeline(thermal_ndarr)
        
    for i in range(len(meta)):
        wkg_meta = meta_df.iloc[i].to_dict()
        file_utils.save_geotiff(os.path.join(args.output, os.path.splitext(wkg_meta['FileName'])[0] + '_thermal.tif'), thermal_ndarr[:,:,i], 'GTiff', wkg_meta)
