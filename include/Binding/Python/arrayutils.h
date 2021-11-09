#ifndef __PYTHON_ARRAY_UTILS_H__
#define __PYTHON_ARRAY_UTILS_H__

typedef enum AXIS {X = 0, Y = 1, Z = 2} axis_t;

PyArrayObject *array_from_stl(const char *filename);

PyArrayObject *vectors_as_RGB(PyArrayObject *vector);

PyArrayObject *normalise(PyArrayObject *arr);

PyArrayObject *rotate(PyArrayObject *arr, double degrees, axis_t axis);

double *rotation_matrix(double degrees, axis_t axis);

#endif