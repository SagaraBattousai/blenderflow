#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API_NAME

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <Binding/Python/arrayutils.h>
#include <Binding/Python/model.h>
#include <Endianness/endianness.h>


#define HEADER_LENGTH 80
#define STL_DIMS 3
#define POINTS_PER_TRIANGLE 3
#define DATA_PER_POINT 3
#define RGB_CHANNELS 3
#define RGB_DIM_COUNT 3
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923
#define ROT_MATRIX_SIZE 9
#define ROT_MATRIX_DIMS 3

static const endianness_t STL_ENDIANNESS = LITTLE;


PyArrayObject *array_from_stl(const char *filename)
{

  FILE *file = fopen(filename, "rb");
  if (file == NULL)
  {
    return NULL;
  }
  //Strip header
  fseek(file, HEADER_LENGTH, SEEK_SET);

  unsigned char triangle_count_data[4];
  unsigned int triangle_count;
  fread(triangle_count_data, 4, 1, file);
  endianness_to_system_cast(STL_ENDIANNESS, triangle_count_data, &triangle_count, sizeof(unsigned int));

  npy_intp const dims[3] = { triangle_count, POINTS_PER_TRIANGLE, DATA_PER_POINT };

  PyArrayObject *polys = (PyArrayObject *) PyArray_SimpleNew(STL_DIMS, dims, NPY_FLOAT32);
  
  if (polys == NULL)
  {
    return NULL;
  }

  unsigned char point_data[4];
  float point;
  for (int i = 0; i < triangle_count; i++)
  {
    //Skip normals
    fseek(file, 12, SEEK_CUR);

    for (int j = 0; j < POINTS_PER_TRIANGLE; j++)
    {
      for (int k = 0; k < DATA_PER_POINT; k++)
      {
        fread(point_data, 4, 1, file);
        endianness_to_system_cast(STL_ENDIANNESS, point_data, &point, sizeof(float));
        float *data = (float *)PyArray_GETPTR3(polys, i, j, k);
        *data = point;
      }
    }
    //Skip attr
    fseek(file, 2, SEEK_CUR);
   }

  return polys;
}

PyArrayObject *vectors_as_RGB(PyArrayObject *vector)
{
  npy_intp *vector_dims = PyArray_DIMS(vector);
  int shape = ceilf(sqrtf(*vector_dims * POINTS_PER_TRIANGLE));

  npy_intp const dims[3] = { shape, shape, RGB_CHANNELS };

  PyArrayObject *RGB = (PyArrayObject *) PyArray_ZEROS(RGB_DIM_COUNT, dims,
    NPY_FLOAT32, 0); // vs SimpleNew and FILLWBYTE ?? Does this steal??

  if (RGB == NULL)
  {
    return NULL;
  }

  float *vector_arr = PyArray_DATA(vector);
  float *RGB_arr = PyArray_DATA(RGB);

  //TODO: Potential error
  //npy_intp indices = (*vector_dims) * (*(vector_dims + 1)) * (*(vector_dims + 2));
  npy_intp indices = PyArray_SIZE(vector); // Safe to use unsafe form as arr is arr

  memcpy(RGB_arr, vector_arr, indices);

  return RGB;
}

PyArrayObject *normalise(PyArrayObject *arr)
{
  PyObject *max = PyArray_Max(arr, NPY_MAXDIMS, NULL); // Must Py_DECREF Okay!
  PyObject *min = PyArray_Min(arr, NPY_MAXDIMS, NULL); // Must Py_DECREF Okay!

  float max_val;
  float min_val;

  PyArray_ScalarAsCtype(max, &max_val);
  PyArray_ScalarAsCtype(min, &min_val);

  float min_abs = fabsf(min_val);

  float scale;

  if (min_abs > max_val)
  {
    scale = min_val;
  }
  else
  {
    scale = max_val;
  }

  npy_intp count = PyArray_SIZE(arr); // Safe to use unsafe form as arr is arr

  float *data = (float *) PyArray_DATA(arr);
  PyArrayObject *norm = (PyArrayObject *) PyArray_NewLikeArray(arr, NPY_CORDER, NULL, 1);
  if (norm == NULL)
  {
    return NULL;
  }

  float *norm_data = (float *)PyArray_DATA(norm);
  for (int i = 0; i < count; i++)
  {
    *(norm_data + i) = ((*(data + i) / scale) + 1) / 2;
  }

  Py_DECREF(max);
  Py_DECREF(min);

  return norm;
}

PyArrayObject *rotate(PyArrayObject *arr, double degrees, axis_t axis)
{

  double *rot_matrix = rotation_matrix(degrees, axis);

  npy_intp const dims[2] = { ROT_MATRIX_DIMS, ROT_MATRIX_DIMS };

  //Dont think I need inc ref etc
  PyObject *rot_array = 
    PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, rot_matrix);

  return  PyArray_MatrixProduct(rot_array, arr);
    
}

//Should be in normal c dll Model but cba to add dll for python build atm
double *rotation_matrix(double degrees, axis_t axis)
{

  double rot_matrix[ROT_MATRIX_SIZE];
  double C = cos(degrees * PI / 180);
  double S = sin(degrees * PI / 180);

  switch (axis)
  {
    case X:
      rot_matrix[0] = 1;
      rot_matrix[1] = 0;
      rot_matrix[2] = 0;

      rot_matrix[3] = 0;
      rot_matrix[4] = C;
      rot_matrix[5] = -S;

      rot_matrix[6] = 0;
      rot_matrix[7] = S;
      rot_matrix[8] = C;
      break;
    case Y:
      rot_matrix[0] = C;
      rot_matrix[1] = 0;
      rot_matrix[2] = S;

      rot_matrix[3] = 0;
      rot_matrix[4] = 1;
      rot_matrix[5] = 0;

      rot_matrix[6] = -S;
      rot_matrix[7] = 0;
      rot_matrix[8] = C;
      break;
    case Z: 
      rot_matrix[0] = C;
      rot_matrix[1] = -S;
      rot_matrix[2] = 0;

      rot_matrix[3] = S;
      rot_matrix[4] = C;
      rot_matrix[5] = 0;

      rot_matrix[6] = 0;
      rot_matrix[7] = 0;
      rot_matrix[8] = 1;
      break;
  }

  return rot_matrix;  
}


