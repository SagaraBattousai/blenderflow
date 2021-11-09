#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#define PY_ARRAY_UNIQUE_SYMBOL MODULE_ARRAY_API_NAME

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <string.h>
#include <Binding/Python/model.h>
#include <Binding/Python/arrayutils.h>

#define FILE_EXTENSION_LENGTH 4

static void Model_dealloc(ModelObject *self)
{
  Py_XDECREF(self->polys);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

//TODO: Maybe neccisary in the future but fornow everything is null
//static PyObject *Model_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
//{
//  ModelObject *self;
//  self = (ModelObject *)type->tp_alloc(type, 0);
//  if (self != NULL)
//  {
//    self->polys = 
//  }
//}

static int Model_init(ModelObject *self, PyObject *args, PyObject *kwds)
{
  PyArrayObject *polys = NULL;
  PyArrayObject *tmp;
  const char *filename;
  int filename_length;

  if (!PyArg_ParseTuple(args, "s#", &filename, &filename_length))
  {
    return -1;
  }

  int extension_offset = filename_length - FILE_EXTENSION_LENGTH;

  const char *file_extension = filename + extension_offset;

  if (strncmp(file_extension, ".stl", 4) == 0)
  {
    polys = array_from_stl(filename);
    if (polys == NULL)
    {
      return -1;
    }
  }
  else
  {
    PyErr_SetString(PyExc_NotImplementedError,
      "Other File types are not yet supported, currently only binary STL's are allowed.");
    return -1;
  }

  tmp = self->polys;
  Py_INCREF(polys);
  self->polys = polys;
  Py_XDECREF(tmp);

  return 0;
}

static PyMemberDef Model_members[] = {
  {"polys", T_OBJECT_EX, offsetof(ModelObject, polys), READONLY, "Polygons"},
  {NULL}
};

static PyObject *
Model_as_RGB(ModelObject *self, PyObject *Py_UNUSED(ignored))
{
  if (self->polys == NULL) {
    PyErr_SetString(PyExc_AttributeError, "polys");
    return NULL;
  }
  
  PyArrayObject *rgb = vectors_as_RGB(self->polys);
  if (rgb == NULL)
  {
    PyErr_SetString(PyExc_AttributeError, "rgb"); //???? TODO: find better exception
    return NULL;
  }
  Py_INCREF(rgb);
  return (PyObject *)rgb;
}


static PyObject *Model_rotate(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
  double degrees = PyLong_AsDouble(args[0]);
  PyObject *error = PyErr_Occurred();
  if (error != NULL)
  {
    PyErr_SetString(error, "Error Converting arg 0 to double");
    return NULL;
  }

  axis_t axis = PyLong_AsLong(args[1]);
  error = PyErr_Occurred();
  if (error != NULL)
  {
    PyErr_SetString(error, "Error Converting arg 1 to integer");
    return NULL;
  }

  /*******************************************************************/
  ModelObject *model = (ModelObject *)self;
  npy_intp *original_dims = PyArray_DIMS(model->polys);

  //i.e. ((triangle count * 3),  3)
  npy_intp new_dims[2] = { original_dims[0] * original_dims[1], original_dims[2] };
  PyArray_Dims new_shape = { .ptr = new_dims, .len = 2 };
  
  PyArrayObject *transform_array = PyArray_Newshape(model->polys, &new_shape, NPY_CORDER);
  transform_array = PyArray_Transpose(transform_array, NULL);

  PyArrayObject *rotated_model = rotate(transform_array, degrees, axis);
  if (rotated_model == NULL)
  {
    PyErr_SetString(PyExc_AttributeError, "rotated_model"); //???? TODO: find better exception
    return NULL;
  }

  rotated_model = PyArray_Transpose(rotated_model, NULL);
  PyArray_Dims original_shape = { .ptr = original_dims, .len = 3 };

  rotated_model = PyArray_Newshape(rotated_model, &original_shape, NPY_CORDER);

  Py_INCREF(rotated_model);
  return rotated_model;
}


static PyMethodDef Model_methods[] = {
    {"as_rgb", (PyCFunction) Model_as_RGB, METH_NOARGS,
     "Returns the model encoded as an rgb float image."},
    {"rotate", (PyCFunction) Model_rotate, METH_FASTCALL,
"rotates model based arrays."},
    {NULL}  /* Sentinel */
};

static PyTypeObject ModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "blenderflow.model.Model",
    .tp_doc = "Model object",
    .tp_basicsize = sizeof(ModelObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_dealloc = (destructor) Model_dealloc,
    .tp_init = (initproc) Model_init,
    .tp_members = Model_members,
    .tp_methods = Model_methods,
};

static PyObject *
blenderflow_normalise(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
  PyArrayObject *arr;

  //Hmmmmmm
  PyArray_OutputConverter(args[0], &arr); //Beautiful exception is automatically thrown
  if (arr == NULL)
  {
    PyErr_SetString(PyExc_AttributeError, "arr"); //???? TODO: find better exception
    return NULL;
  }

  PyArrayObject *norm = normalise(arr);
  
  if (norm == NULL)
  {
    PyErr_SetString(PyExc_AttributeError, "norm"); //???? TODO: find better exception
    return NULL;
  }

  Py_INCREF(norm);
  return (PyObject *) norm; // arr;
}

static PyMethodDef blenderflow_methods[] = {
  {"normalise",  (PyCFunction)blenderflow_normalise, METH_FASTCALL,
     "Normalises model based arrays."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef blenderflow = {
    PyModuleDef_HEAD_INIT,
    .m_name = "blenderflow",
    .m_doc = "Blenderflow extension module for handling models as numpy arrays",
    .m_size = -1,
    .m_methods = blenderflow_methods,
};

PyMODINIT_FUNC PyInit_blenderflow(void)
{
  PyObject *m;
  if (PyType_Ready(&ModelType) < 0)
  {
    return NULL;
  }

  m = PyModule_Create(&blenderflow);
  if (m == NULL)
  {
    return NULL;
  }

  Py_INCREF(&ModelType);
  if (PyModule_AddObject(m, "Model", (PyObject *)&ModelType) < 0)
  {
    Py_DECREF(&ModelType);
    Py_DECREF(m);
    return NULL;
  }

  import_array();
  return m;

}