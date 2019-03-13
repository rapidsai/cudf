# Copyright (c) 2018, NVIDIA CORPORATION.

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import versioneer
from distutils.sysconfig import get_python_lib


install_requires = [
    'numba',
    'cython'
]

cython_files = ['cudf/bindings/*.pyx']

extensions = [
    Extension("*",
              sources=cython_files,
              include_dirs=[
                '../cpp/include/',
                '../cpp/thirdparty/dlpack/include/dlpack/'
              ],
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
      ],
      # Include the separately-compiled shared library
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      ext_modules=cythonize(extensions),
      packages=find_packages(include=['cudf', 'cudf.*']),
      package_data={
        'cudf.tests': ['data/*.pickle'],
      },
      install_requires=install_requires,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False
      )
