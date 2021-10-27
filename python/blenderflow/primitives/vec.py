import ctypes

__all__ = ["VERTEX", "POINT", "Vertex", "Point"] 

class VERTEX(ctypes.Structure):
  _fields_ = [("x", ctypes.c_float),
              ("y", ctypes.c_float),
              ("z", ctypes.c_float)]

class POINT(ctypes.Structure):
  _fields_ = [("x", ctypes.c_float),
              ("y", ctypes.c_float)]

class Vertex:
  def __init__(self, x=0.0, y=0.0, z=0.0):
    self.x = x
    self.y = y
    self.z = z
    self._as_parameter_ = VERTEX(x, y, z)

  @classmethod
  def from_c(cls, c_vertex):
    return cls(c_vertex.x, c_vertex.y, c_vertex.z)

  def to_c(self):
    return VERTEX(self.x, self.y, self.z)

  def as_list(self):
    return [self.x, self.y, self.z]

  def __repr__(self):
    return f"Vertex({self.x}, {self.y}, {self.z})"

  def swizel(self, swizel_vector=(0,2,1)):
    list_form = self.as_list()
    self.x = list_form[swizel_vector[0]]
    self.y = list_form[swizel_vector[1]]
    self.z = list_form[swizel_vector[2]]

  def orthographic_projection(self):
    return Point(self.x, self.y)

  def perspective_projection(self, f):
    return Point(self.x * f / self.z, self.y * f / self.z)

class Point:
  def __init__(self, x=0.0, y=0.0):
    self.x = x
    self.y = y
    self._as_parameter_ = POINT(x, y)

  @classmethod
  def from_c(cls, c_point):
    return cls(c_point.x, c_point.y)

  def to_c(self):
    return POINT(self.x, self.y)

  def as_list(self):
    return [self.x, self.y]

  def __repr__(self):
    return f"Point({self.x}, {self.y})"

  def swizel(self, swizel_vector=(1,0)):
    list_form = self.as_list()
    self.x = list_form[swizel_vector[0]]
    self.y = list_form[swizel_vector[1]]
