from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

import os
import versioneer



packages = ['cudf',
            'cudf.bind',
            'cudf.tests'
            ]

install_requires = [
    'numba',
]

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

cython_files = ['cudf/bind/*.pyx']

extensions = [
    Extension("*",
              sources=cython_files,
              include_dirs=[numpy_include, '/home/dante/Projects/git/cudf/cython/libgdf/include/gdf'],
              library_dirs=['/home/dante/miniconda3/envs/cython/lib/'],
              libraries=['gdf'],
              language='c++',
              extra_compile_args=['-std=c++11'],
              runtime_library_dirs=['/usr/local/lib'])
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


# setup(name=projectName,
#       version=__version__,
#       license=__license__,
#       description=shortDescription,
#       long_description=longDescription,
#       author=__author__,
#       author_email=__email__,
#       setup_requires=['cython'],
#       ext_modules=build_extension(),
#       cmdclass=cmdclass,
#       packages=find_packages(),
#       classifiers=classifiers
#       )
