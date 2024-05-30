# Installation
This script should work with most recent Python versions, although it has not been tested extensively on them. Windows support is unknown.

 Please first install the requirements by running 
> pip install -r requirements.txt

Then install the Python version of the GDAL library matching the version of your system install of GDAL. The version can be found and installed by running 
> pip install --force-reinstall GDAL[numpy]==$(gdal-config --version)

Also please ensure that a recent version of exiftool is installed, ideally greater than version 12, although the exact minimum required version is unknown.

# Cookbook
Example scenarios:

As is, this script will always try to extract a visible image and thermal image from the provided files.

A.1) convert a directory of flir rjpeg to tiffs w/temperatures (vue pro r on a tower) - no georef. 
> python3 process.py -i [input_dir] -o [output_dir] --height [height/distance in meters]

Exactly 1 elevation referencing parameter must be provided. Height is the simplest and does not rely on using any metadata in its calculations.

A.2) in addition to A.1, also extract the vis to jpegs (duo pro r on a tower or big drone) - Using geo referencing from the photo's exif GPS tags against a fixed terrain elevation. 
> python3 process.py -i [input_dir] -o [output_dir] --elevation [elevation in meters]

B) Convert a directory of flir rjpeg to tiffs and associate dji gps tags from a matched set of photos from the DJI drone. (vue pro r on a drone). This should also be easy to extend in the future for other potential location services.
> python3 process.py -i [input_dir] -o [output_dir] --dji_gps_source [dji_photo_dir] --elevation [elevation in meters]

C) convert a directory of tiffs to temperature (tower situation) and modify temperature conversion parameters (perhaps from the tower logger info?). 
> python3 process.py -i [input_dir] -o [output_dir] --elevation [elevation in meters] --ground_station_log [path_to_logfile] --groud_station_cols [timestamp_col_name],[atmos_temp_col_name],[humidity_col_name]

Processing can be further modified by providing an emissivity value for the observed surface using the `-e` flag.

> python3 process.py -i [input_dir] -o [output_dir] --height [height/distance in meters] -e [float_emissive_property]

D) Convert a directory of DJI thermal images to tiffs. The thermal images should not be mixed with visible images in the same directory. All other options should also work with the DIRP option.
> python3 process.py -i [input_dir] -o [output_dir] --height [height/distance in meters] --dirp