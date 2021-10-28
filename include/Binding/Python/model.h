#ifndef __PYTHON_MODEL_H__
#define __PYTHON_MODEL_H__

#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
  PyObject_HEAD
  PyObject *polys;
} ModelObject;

#endif