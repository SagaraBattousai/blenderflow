import ctypes
from .vec import * #Vertex, Point, VERTEX, POINT
__all__ = ["TRIANGLE3D", "TRIANGLE2D", "Triangle3D", "Triangle2D"]


class TRIANGLE3D(ctypes.Structure):
  _fields_ = [("i", VERTEX),
              ("j", VERTEX),
              ("k", VERTEX)]

class TRIANGLE2D(ctypes.Structure):
  _fields_ = [("i", POINT),
              ("j", POINT),
              ("k", POINT)]


class Triangle3D:
  def __init__(self, i=None, j=None, k=None):
    self.i = i
    self.j = j
    self.k = k
    self._as_parameter_ = TRIANGLE3D(i.to_c(),
                                     j.to_c(),
                                     k.to_c())
  @classmethod
  def from_c(cls, c_triangle):
    i = Vertex.from_c(c_triangle.i)
    j = Vertex.from_c(c_triangle.j)
    k = Vertex.from_c(c_triangle.k)
    return cls(i, j, k)

  def as_list(self):
    return [self.i.as_list(), self.j.as_list(), self.k.as_list()]

  def __repr__(self):
    return f"Triangle3D(\n\t{self.i}, \n\t{self.j}, \n\t{self.k}\n\t)"

class Triangle2D:
  def __init__(self, i=None, j=None, k=None):
    self.i = i
    self.j = j
    self.k = k
    self._as_parameter_ = TRIANGLE2D(i.to_c(),
                                     j.to_c(),
                                     k.to_c())
  @classmethod
  def from_c(cls, c_triangle):
    i = Point.from_c(c_triangle.i)
    j = Point.from_c(c_triangle.j)
    k = Point.from_c(c_triangle.k)
    return cls(i, j, k)

  def as_list(self):
    return [self.i.as_list(), self.j.as_list()]

  def __repr__(self):
    return f"Triangle2D({self.i}, {self.j})"

