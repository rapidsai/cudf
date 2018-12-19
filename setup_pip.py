# Copyright (c) 2018, NVIDIA CORPORATION.
# This setup.py packages librmm, libcudf, libgdf_cffi, librmm_cffi, and cudf all together for distribution via pypi
# cython & cffi must be installed to run this
import os
import subprocess
import shutil
import sys
import sysconfig
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import glob
from distutils.command.install_headers import install_headers
from distutils.sysconfig import get_python_lib
from Cython.Build import cythonize
import numpy

sys.path.append('cpp/python')
os.environ['RMM_HEADER'] = 'cpp/src/rmm/memory.h'
os.environ['CUDF_INCLUDE_DIR'] = 'cpp/include/cudf'

from libgdf_cffi import libgdf_build
from librmm_cffi import librmm_build


CMAKE_EXE=os.environ.get('CMAKE_EXE', shutil.which('cmake'))

def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return os.path.join('build', f.format(dirname=dname,
                                          platform=sysconfig.get_platform(),
                                          version=sys.version_info))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildExt(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            output_dir = os.path.abspath(
                os.path.dirname(self.get_ext_fullpath(ext.name)))

            build_type = 'Debug' if self.debug else 'Release'
            cmake_args = [CMAKE_EXE,
                          ext.sourcedir,
                          '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + output_dir,
                          '-DCMAKE_BUILD_TYPE=' + build_type]
            cmake_args.extend(
                [x for x in
                 os.environ.get('CMAKE_COMMON_VARIABLES', '').split(' ')
                 if x])

            env = os.environ.copy()
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(cmake_args,
                                  cwd=self.build_temp,
                                  env=env)
            subprocess.check_call(['make', '-j', ext.name],
                                  cwd=self.build_temp,
                                  env=env)
            print()
        else:
            super().build_extension(ext)


class InstallHeaders(install_headers):
    """
    This overrides the install_header command to walk a directory tree and copy all of the specified headers
    This is necessary because the built-in install_headers installs all headers in the same directory
    """

    def run(self):
        headers = self.distribution.headers
        if not headers:
            return
        self.mkpath(self.install_dir)
        for header in headers:
            for dirpath, dirnames, filenames in os.walk(header):
                for file in filenames:
                    if file.endswith('.h'):
                        path = os.path.join(dirpath, file)
                        install_target = os.path.join(self.install_dir,
                                                      path.replace(header, ''))
                        self.mkpath(os.path.dirname(install_target))
                        (out, _) = self.copy_file(path, install_target)
                        self.outfiles.append(out)


# setup does not clean up the build directory, so do it manually
shutil.rmtree('build', ignore_errors=True)

install_requires = [
    'pandas>=0.20,<0.21',
    'numba>=0.40.0dev,<0.41',
    'pycparser==2.19',
    'pyarrow>=0.10,<0.11',
    'cffi>=1.0.0',
    'cython>=0.28,<0.29',
    'nvstrings'
]

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

cython_files = ['python/cudf/bindings/*.pyx']

extensions = [
    CMakeExtension('rmm', sourcedir='cpp'),
    CMakeExtension('cudf', sourcedir='cpp'),
    Extension("*",
              sources=cython_files,
              include_dirs=[numpy_include, 'cpp/include/'],
              library_dirs=[get_python_lib(), distutils_dir_name('lib')],
              libraries=['cudf'],
              language='c++',
              extra_compile_args=['-std=c++11'])
]

packages = [
    'libgdf_cffi',
    'librmm_cffi',
]
packages += find_packages(where='python')

setup(name='cudf',
      description='cuDF - GPU Dataframe',
      version=os.environ.get('GIT_DESCRIBE_TAG', '0.0.0.dev').lstrip('v'),
      classifiers=[
          "Intended Audience :: Developers",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.5",
          ],
      author="NVIDIA Corporation",
      packages=packages,
      package_dir={
          'cudf': 'python/cudf',
          'libgdf_cffi': 'cpp/python/libgdf_cffi',
          'librmm_cffi': 'cpp/python/librmm_cffi'
      },
      package_data={
          'cudf.tests': ['data/*'],
      },
      install_requires=install_requires,
      setup_requires=install_requires,
      cffi_modules=[
          'cpp/python/libgdf_cffi/libgdf_build.py:ffibuilder',
          'cpp/python/librmm_cffi/librmm_build.py:ffibuilder'
      ],
      ext_modules=cythonize(extensions),
      cmdclass={
          'build_ext': CMakeBuildExt,
          'install_headers': InstallHeaders
      },
      headers=['cpp/include/'],
      zip_safe=False
      )
