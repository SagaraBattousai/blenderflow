import ctypes
from ..primitives.vec import VERTEX

__all__ = ["PLY_MODEL"]

class PLY_MODEL(ctypes.Structure):
  _fields_ = [("vertex_count", ctypes.c_size_t),
              ("face_count", ctypes.c_size_t),
              ("endian", ctypes.c_int),
              ("verticies", ctypes.POINTER(VERTEX)),
              ("face_indicies", ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))]

