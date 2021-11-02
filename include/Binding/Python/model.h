#ifndef __PYTHON_MODEL_H__
#define __PYTHON_MODEL_H__

typedef struct {
  PyObject_HEAD
  PyArrayObject *polys;
} ModelObject;

#endif