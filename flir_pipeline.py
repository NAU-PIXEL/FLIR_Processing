import importlib
import os
import sys

class Pipeline():
  def __init__(self, user_fn_path="user_pipeline"):
    self.fn_dict = dict()
    self.fn_help_dict = dict()
  
  # loads pipelines, with the expectation that the pipeline imports this module
  # itself, and then uses the add_pipeline_function_dec decorator on each
  # function that they want to add to the pipeline
  def _post_init(self):
    importlib.import_module('flir_pipeline_functions')
    for file in os.listdir('user_pipeline'):
      # remove ext
      file = os.path.splitext(file)[0]
      importlib.import_module(f'user_pipeline.{file}')

  def add_pipeline_function_dec(self, name, help=None):
    def decorator(pipeline_fn):
      if name in self.fn_dict:
        raise KeyError("Function already exists, not overwriting.")
      self.fn_dict[name] = pipeline_fn
      if help:
        self.fn_help_dict[name] = help
      return pipeline_fn
    return decorator

  def verify_pipeline_fns(self, to_check):
    for fn in to_check:
      if fn not in self.fn_dict:
        return False
    return True
  
  def build_pipeline(self, fn_list, opts=None):
    def compose(inside, outside):
      return lambda ndarr: inside(outside(ndarr))
    base_pipeline = lambda arr: arr
    for fn_name in fn_list:
      fn = self.fn_dict[fn_name]
      base_pipeline = compose(base_pipeline, fn)
    return base_pipeline

  def pipeline_help(self):
    for fn in self.fn_dict:
      if fn in self.fn_help_dict:
        print(fn, self.fn_help_dict[fn])
      else:
        print(fn)

_pipeline = Pipeline()

# we have to do this weird post init hook thing because we cannot load the user
# modules while this module is initializing, since the users need to import 
# this module itself. 
# we resolve this using this __getattr__ override, since accessing the pipeline
# attribute will happen after the module has initialized, but still as early as possible.
# this allows consumers of the pipeline functions to just write
# `import flir_pipeline` and then use `flir_pipeline.pipeline.build_pipeline`
# without having to have extra code doing `flir_pipeline.pipeline._post_init()`
def __getattr__(attr):
  if attr == "pipeline":
    _pipeline._post_init()
    setattr(sys.modules[__name__], 'pipeline', _pipeline)
    return _pipeline