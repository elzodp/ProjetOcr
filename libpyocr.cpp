#include <iostream>
#include "test_xor.h"
#include "NN2.h"

#include <Python.h>
// Si le <Python.h> est introuvable :
// sudo apt install python3.8-dev
// locate Python.h
// sudo ln -sv /usr/include/python3.8/* /usr/include/

using real = NN2::real;
using vec = NN2::vec;
using mat = NN2::mat;

static PyObject *OCR_error = NULL;
// Site sur comment déclarer les fonctions en Pythons etc pour une interface
// https://www.tutorialspoint.com/python/python_further_extensions.htm
//
//


// Créer une table qui puisse harger les fichiers en python


//On initialise le bon set up
static PyObject *OCR_version(PyObject *self)
{
  return Py_BuildValue("s", "OCR version 0.X");
	// le s représente le format, le s : Converts a null-terminated C string to a Python object. If the C string pointer is NULL, None is returned.
}
// /!\ Py_BuildValue va créer une nouvelle référence vers l'argument en question je crois

static PyObject *OCR_sigmoid(PyObject *self, PyObject *args)
{
  real x = 0.0;

  if (!PyArg_ParseTuple(args, "d", &x))
	// le f permet de transformer les floats de Python en en float C/C++, l'inverse de BuildValue en qq sorte
    return NULL;
  return Py_BuildValue("d",NN2::sigmoid(x));
  // Tout marche normalemen, on part de python, on a notre float en C, puis on calcule le sigmoid puis on retourne la valeur en float python
}

static PyObject *OCR_d_sigmoid(PyObject *self, PyObject *args)
{
  real x = 0.0;

  if (!PyArg_ParseTuple(args, "d", &x))
    return NULL;

  return Py_BuildValue("d", NN2::d_sigmoid(x));
}

static PyMethodDef OCR_functions[] = {

  {"sigmoid",OCR_sigmoid,METH_VARARGS,"Valeur sigmoid d'un flottant"},
  {"d_sigmoid",OCR_d_sigmoid,METH_VARARGS,"La dérivé de la sigmoid du flottant"},
  {"version",(PyCFunction)OCR_version,METH_VARARGS,"Version du Projet" },

  { NULL, NULL, 0, NULL}
};

// Appel depuis Python qui crée un objet en C (NN), on entraîne  depuis l'interface et l'utiliser


//

static PyModuleDef OCR_module = {

  PyModuleDef_HEAD_INIT,
  "OCR",
  "OCR Python Interface",
  -1,
  OCR_functions
};

PyMODINIT_FUNC PyInit_OCR()
{
  PyObject *obj = PyModule_Create(&OCR_module);

  if (!obj)
    return NULL;

  OCR_error = PyErr_NewException("OCR.error", NULL, NULL);
  Py_XINCREF(OCR_error);

  if (PyModule_AddObject(obj, "error", OCR_error) < 0)
    {
      Py_XDECREF(OCR_error);
      Py_CLEAR(OCR_error);
      Py_DECREF(obj);
      return NULL;
    }

  return obj;
}
