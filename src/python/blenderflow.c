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
  PyObject *tmp;
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
  
  return vectors_as_RGB(self->polys);
}

static PyMethodDef Model_methods[] = {
    {"as_rgb", (PyCFunction) Model_as_RGB, METH_NOARGS,
     "Returns the model encoded as an rgb float image."
    },
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

static PyModuleDef blenderflowmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "blenderflow",
    .m_doc = "Blenderflow extension module for handling models as numpy arrays",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_blenderflow(void)
{
  PyObject *m;
  if (PyType_Ready(&ModelType) < 0)
  {
    return NULL;
  }

  m = PyModule_Create(&blenderflowmodule);
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