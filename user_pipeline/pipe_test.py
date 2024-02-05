from flir_pipeline import pipeline

@pipeline.add_pipeline_function_dec('addn')
def add_n(image_ndarr, n, extra=0):
  return image_ndarr + n + extra