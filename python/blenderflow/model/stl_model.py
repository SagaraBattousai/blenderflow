import ctypes
import collections.abc
import numpy as np
import matplotlib.pyplot as plt
from ..primitives.vec import *
from ..primitives.tri import *
from .. import _model as model

__all__ = ["STL_MODEL","stl_model"]


#ALIAS
X = 0
Y = 1
Z = 2


class STL_MODEL(ctypes.Structure):
  _fields_ = [("triangle_count", ctypes.c_uint),
              ("normals", ctypes.POINTER(VERTEX)),
              ("triangles", ctypes.POINTER(TRIANGLE3D))]

class stl_model(collections.abc.Sequence):
  def __init__(self, filename=None, c_stl_model=None):
    if c_stl_model is None:
      self.c_stl_model = STL_MODEL()
      model.get_stl_model(ctypes.byref(self.c_stl_model), filename)
    else:
      self.c_stl_model = c_stl_model
    self._as_parameter_ = ctypes.byref(self.c_stl_model)

  def _get_index(self, key):
    length = self.c_stl_model.triangle_count
    if key >= length or key < -1 * length:
      raise ValueError("Key is outside range of values")
    else:    
      return length + key if key < 0 else key

  def __len__(self):
    return self.c_stl_model.triangle_count

  def __getitem__(self, key):
    # length = self.c_stl_model.triangle_count
    if isinstance(key, int):
      index = self._get_index(key)
      normal = self.c_stl_model.normals[index]
      triangle = self.c_stl_model.triangles[index]

      return (Vertex.from_c(normal),
              Triangle3D.from_c(triangle))

    elif isinstance(key, slice):
      #TODO
      return self.__getitem__(key.start)
    else:
      raise TypeError(
          f"Key must be one of 'int' or 'slice' but had type {type(key)}"
          )


  def get_triangles_as_list(self):
    tris = []
    for triangle_index in range(self.c_stl_model.triangle_count):
      tris.extend(Triangle3D.from_c(self.c_stl_model.triangles[triangle_index]).as_list())
    return tris

  def get_perspective(self, f=0):
    xyz = np.array(self.get_triangles_as_list())
    if f == 0:
      tri_points = [[a[X], 0, a[Z]] for a in xyz]
    else:
      tri_points = [[f * (a[X] / a[Y]), f, f * (a[Z] / a[Y])] for a in xyz]

    return tri_points 








