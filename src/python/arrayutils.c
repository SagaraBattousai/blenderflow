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
  printf("%i, %i\n", shape, *vector_dims);

  npy_intp const dims[3] = { shape, shape, RGB_CHANNELS };

  PyArrayObject *RGB = (PyArrayObject *) PyArray_ZEROS(RGB_DIM_COUNT, dims,
    NPY_FLOAT32, 0); // vs SimpleNew and FILLWBYTE ?? Does this steal??

  if (RGB == NULL)
  {
    return NULL;
  }

  float *vector_arr = PyArray_DATA(vector);
  float *RGB_arr = PyArray_DATA(RGB);
  npy_intp indices = (*vector_dims) * (*(vector_dims + 1)) * (*(vector_dims + 2));
  for (npy_intp i = 0; i < indices; i++) // Could I memset/memcopy data over?
  {
    *(RGB_arr + i) = *(vector_arr + i);
  }
  return RGB;
}

