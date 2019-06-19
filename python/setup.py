# Copyright (c) 2018, NVIDIA CORPORATION.

import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import versioneer
from distutils.sysconfig import get_python_lib


install_requires = [
    'numba',
    'cython'
]

cython_files = [
    'cudf/bindings/**/*.pyx',
]


cuda_include_dir = '/usr/local/cuda/include'
cuda_lib_dir = "/usr/local/cuda/lib"

if os.environ.get('CUDA_HOME', False):
    cuda_lib_dir = os.path.join(os.environ.get('CUDA_HOME'), 'lib64')
    cuda_include_dir = os.path.join(os.environ.get('CUDA_HOME'), 'include')


rmm_include_dir = '/include'
rmm_lib_dir = '/lib'

if os.environ.get('CONDA_PREFIX', None):
    conda_prefix = os.environ.get('CONDA_PREFIX')
    rmm_include_dir = conda_prefix + rmm_include_dir


extensions = [
    Extension("*",
              sources=cython_files,
              include_dirs=[
                '../cpp/include/cudf',
                '../cpp/thirdparty/dlpack/include/dlpack/',
                '../cpp/thirdparty/rmm/include/',
                '../cpp/thirdparty/rmm/thirdparty/cnmem/include',
                rmm_include_dir,
                cuda_include_dir
              ],
              library_dirs=[get_python_lib()],
              libraries=['cudf'],
              language='c++',
              extra_compile_args=['-std=c++14'])
]

setup(name='cudf',
      description="cuDF - GPU Dataframe",
      version=versioneer.get_version(),
      classifiers=[
        # "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        # "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
      ],
      # Include the separately-compiled shared library
     author="NVIDIA Corporation", setup_requires=['cython'],
      ext_modules=cythonize(extensions),
      packages=find_packages(include=['cudf', 'cudf.*']),
      install_requires=install_requires,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False
      )
