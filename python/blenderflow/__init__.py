import sys
import ctypes

def import_models():
  shared_library_name = "./lib/model"
  if sys.platform == "win32":
    shared_library_name += ".dll"
  else:
    shared_library_name += ".so"
  return ctypes.CDLL(shared_library_name)

_model = import_models()

from .primitives import *
from .model import *


if __name__ == "__main__":
  models = import_models()
  i = stl_model()
  #print(i.triangels.contents())
  models.get_stl_model(i, b"Colon.stl")

  tc = i.c_stl_model.triangle_count
  
  i.c_stl_model.triangels = (TRIANGLE3D * tc)()

  models.get_stl_model(i, b"Colon.stl")

  j = i.c_stl_model.triangels
  print(j.contents.i.x)
  print(j[5].i.x)
