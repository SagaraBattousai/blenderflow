#ifndef __PYTHON_ARRAY_UTILS_H__
#define __PYTHON_ARRAY_UTILS_H__

PyArrayObject *array_from_stl(const char *filename);

PyArrayObject *vectors_as_RGB(PyArrayObject *vector);

#endif