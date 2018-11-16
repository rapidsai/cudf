# Copyright (c) 2018, NVIDIA CORPORATION.

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

import versioneer
from distutils.sysconfig import get_python_lib


packages = ['cudf',
            'cudf.bindings',
            'cudf.tests'
            ]

install_requires = [
    'numba',
    'cython'
]

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

cython_files = ['cudf/bindings/*.pyx']

extensions = [
    Extension("*",
              sources=cython_files,
              include_dirs=[numpy_include, '../cpp/include/'],
              library_dirs=[get_python_lib()],
              libraries=['cudf'],
              language='c++',
              extra_compile_args=['-std=c++11'])
]

setup(name='cudf',
      description="cuDF - GPU Dataframe",
      version=versioneer.get_version(),
      classifiers=[
        # "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        # "Operating System :: OS Independent",
        "Programming Language :: Python",
        # "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
      ],
      # Include the separately-compiled shared library
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      ext_modules=cythonize(extensions),
      packages=packages,
      package_data={
        'cudf.tests': ['data/*.pickle'],
      },
      install_requires=install_requires,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False
      )
