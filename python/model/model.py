import sys
import ctypes

def import_models():
  shared_library_name = "./lib/model"
  if sys.platform == "win32":
    shared_library_name += ".dll"
  else:
    shared_library_name += ".so"
  return ctypes.CDLL(shared_library_name)


class VERTEX(ctypes.Structure):
  _fields_ = [("x", ctypes.c_float),
              ("y", ctypes.c_float),
              ("z", ctypes.c_float)]


class POINT(ctypes.Structure):
  _fields_ = [("x", ctypes.c_float),
              ("y", ctypes.c_float)]

class TRIANGLE3D(ctypes.Structure):
  _fields_ = [("i", VERTEX),
              ("j", VERTEX),
              ("k", VERTEX)]

class TRIANGLE2D(ctypes.Structure):
  _fields_ = [("i", POINT),
              ("j", POINT),
              ("k", POINT)]

class STL_MODEL(ctypes.Structure):
  _fields_ = [("triangle_count", ctypes.c_uint),
              ("normals", ctypes.POINTER(VERTEX)),
              ("triangels", ctypes.POINTER(TRIANGLE3D))]

class PLY_MODEL(ctypes.Structure):
  _fields_ = [("vertex_count", ctypes.c_size_t),
              ("face_count", ctypes.c_size_t),
              ("endian", ctypes.c_int),
              ("verticies", ctypes.POINTER(VERTEX)),
              ("face_indicies", ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))]

class  stl_model:
  def __init__(self):
    self.c_stl_model = STL_MODEL()
    self._as_parameter_ = ctypes.byref(self.c_stl_model)

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
