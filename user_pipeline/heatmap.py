from flir_pipeline import pipeline

import cv2
import numpy as np

def normalize(array, rescaled_min=0, rescaled_max=255):
  normalized = (array - array.min()) / (array.max() - array.min())
  
  return normalized * (rescaled_max - rescaled_min) + rescaled_min

@pipeline.add_pipeline_function_dec('heatmap')
def heatmap(image_ndarr):
  # normalize input ndarry so that when it's taken to U8 it has values from 0 to 255
  normalized = normalize(image_ndarr).astype(np.uint8)
  
  # use cv2 to create a heatmap
  # apply color map over each 2D greyscale image
  colormap_accumulator = []
  for i in range(normalized.shape[2]):
    heat_img = cv2.applyColorMap(normalized[:, :, i], cv2.COLORMAP_HOT)
    # reorganize from BGR to RGB
    heat_img = np.flip(heat_img, axis=2)
    colormap_accumulator.append(heat_img)
    colormap_accumulator[i] = np.concatenate(
      [colormap_accumulator[i], np.expand_dims(image_ndarr[:,:,i], axis=2)],
      axis=2)

  # stack the color maps
  heatmap = np.stack(colormap_accumulator, axis=2)
  
  return heatmap  