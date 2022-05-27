from distutils.core import setup, Extension
import os


OCR = Extension("OCR", sources = [ "libpyocr.cpp", "../NN2.cpp", "../tests/test_xor.cpp" ])

setup(name        = "OCR",
      version     = "0.X",
      description = "OCR in C/C++ with Py Extension",
      include_dirs = ["../../include"],
      ext_modules = [OCR])
