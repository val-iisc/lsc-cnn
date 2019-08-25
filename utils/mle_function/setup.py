"""
setup.py: Driving code for MLE function
Authors       : mns
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "error_function",
    ext_modules = cythonize("error_function.pyx", include_path = [numpy.get_include()]),
)

