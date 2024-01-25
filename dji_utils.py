import os

import exiftool
import pandas as pd

def gps_table_from_images(source_path, offset):
  images = os.listdir(source_path)
  images = [os.path.join(source_path, image) for image in images]
  
  exif = exiftool.ExifToolHelper()
  meta = exif.get_metadata(images)

  original_cols = ['SourceFile', 'EXIF:DateTimeOriginal', 'EXIF:GPSLatitude', 'EXIF:GPSLongitude', 'EXIF:GPSAltitude']

  rename_cols = ['filename', 'datetime', 'latitude', 'longitude', 'altitude']

  ew_mult = {'E': 1, 'W':-1}
  ns_mult = {'N': 1, 'S':-1}

  df = pd.DataFrame(meta)
  # correct sign of GPS decimal
  df['EXIF:GPSLatitude'] = df['EXIF:GPSLatitude'] * df['EXIF:GPSLatitudeRef'].map(ns_mult)
  df['EXIF:GPSLongitude'] = df['EXIF:GPSLongitude'] * df['EXIF:GPSLongitudeRef'].map(ew_mult)
  # take only cols we want
  df = df[original_cols]
  # rename
  df.columns = rename_cols
  # convert datetime column from string to datetime
  # DJI uses weird format, 2024:01:18 20:08:10, no timezone.
  df['datetime'] = pd.to_datetime(df['datetime'], format="%Y:%m:%d %H:%M:%S")
  df['datetime'] = df['datetime'] + pd.Timedelta(seconds=offset)
  return df