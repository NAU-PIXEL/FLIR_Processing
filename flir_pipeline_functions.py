import numpy as np
import scipy.ndimage

from flir_pipeline import pipeline

destripe_help = """Destripe image using moving mean along striped axis method"""

@pipeline.add_pipeline_function_dec("destripe", help=destripe_help)
def destripe_rows(image_ndarr):
  row_avgs = np.mean(image_ndarr, axis=1)
  # find what the equivalent fill value is
  conv = scipy.ndimage.uniform_filter1d(row_avgs, 31,)
  conv = row_avgs - conv
  # have to do this since we collapsed a middle dimension to get it back so we 
  # have broadcastable shapes again.
  conv = np.expand_dims(conv, axis=1)
  image_ndarr = image_ndarr - conv
  
  return image_ndarr

ffc_s_help = """Perform flat field correction using a mean of the processed images. Assumes that across the entire image sequence the average of the images should be evenly bright, so any extra darkness on the edges is the artefact that must be corrected."""

@pipeline.add_pipeline_function_dec("ffc_s", help=ffc_s_help)
def static_flat_field_correct(image_ndarr):
  # filter each layer with box filter, in reference code it's 191 pix squared
  filtered = scipy.ndimage.uniform_filter(image_ndarr, size=191, mode="nearest", axes=(0,1))

  # get 10 pix squared section of each filtered image, then take mean of that subsection
  mid_x, mid_y = filtered.shape[0] // 2, filtered.shape[1] // 2
  center_temps = filtered[mid_x - 5:mid_x+5, mid_y - 5:mid_y+5, :]
  center_means = np.mean(center_temps, axis=(0,1))

  # get mean of each pixel across every input image
  flat = np.mean(filtered - center_means, axis=2)

  # since flat is 2D and image_ndarr is 3D we have to make them the same dimensions
  image_ndarr = image_ndarr - flat[..., np.newaxis]

  # this is modified in place but also return the reference for niceness of code.
  return image_ndarr
