# Copyright (c) 2018, NVIDIA CORPORATION.
# This setup.py packages these modules together as one pip package:
#   * librmm
#   * libcudf
#   * libgdf_cffi
#   * librmm_cffi
#   * cudf

# cython, cffi, & nvstrings must be installed to run this
# conda create -n build python=X.Y cython cffi -c nvidia nvstrings
# python setup_pip.py bdist_wheel

import os
import subprocess
import shutil
import sys
import sysconfig
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.command.install_headers import install_headers
from distutils.sysconfig import get_python_lib
from Cython.Build import cythonize
import zipfile
from hashlib import sha256
from base64 import urlsafe_b64encode

os.environ['RMM_HEADER'] = 'cpp/thirdparty/rmm/include/rmm/rmm_api.h'
os.environ['CUDF_INCLUDE_DIR'] = 'cpp/include/cudf'

CMAKE_EXE = os.environ.get('CMAKE_EXE', shutil.which('cmake'))
if not CMAKE_EXE:
    print('cmake executable not found. '
          'Set CMAKE_EXE environment or update your path')
    sys.exit(1)


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
    The built-in install_headers installs all headers in the same directory
    This overrides that to walk a directory tree and copy the tree of headers

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

cuda_version = ''.join(os.environ.get('CUDA', 'unknown').split('.')[:2])

install_requires = [
    'pandas>=0.23.4',
    'numba>=0.40.0,<0.42',
    'pycparser==2.19',
    'pyarrow==0.12.1',
    'cffi>=1.0.0',
    'cython>=0.29,<0.30',
    'numpy>=1.14'
]
setup_requires = install_requires.copy()

install_requires.append('nvstrings-cuda{}'.format(cuda_version))

cython_files = ['python/cudf/bindings/*.pyx']

extensions = [
    CMakeExtension('rmm', sourcedir='cpp'),
    CMakeExtension('cudf', sourcedir='cpp'),
    Extension("*",
              sources=cython_files,
              include_dirs=['cpp/include/', 'thirdparty/dlpack/include/dlpack'],
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

name = 'cudf-cuda{}'.format(cuda_version)
version = os.environ.get('GIT_DESCRIBE_TAG', '0.0.0.dev0').lstrip('v')
setup(name=name,
      description='cuDF - GPU Dataframe',
      long_description=open('README.md', encoding='UTF-8').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/rapidsai/cudf',
      version=version,
      classifiers=[
          "Intended Audience :: Developers",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7"
      ],
      author="NVIDIA Corporation",
      license='Apache 2.0',
      packages=packages,
      package_dir={
          'cudf': 'python/cudf',
          'libgdf_cffi': 'cpp/python/libgdf_cffi',
          'librmm_cffi': 'cpp/thirdparty/rmm/python/librmm_cffi'
      },
      package_data={
          'cudf.tests': ['data/*'],
      },
      install_requires=install_requires,
      setup_requires=setup_requires,
      python_requires='>=3.6,<3.8',
      cffi_modules=[
          'cpp/python/libgdf_cffi/libgdf_build.py:ffibuilder',
          'cpp/thirdparty/rmm/python/librmm_cffi/librmm_build.py.in:ffibuilder'
      ],
      ext_modules=cythonize(extensions),
      cmdclass={
          'build_ext': CMakeBuildExt,
          'install_headers': InstallHeaders
      },
      headers=['cpp/include/'],
      zip_safe=False
      )


def convert_to_manylinux(name, version):
    """
    Modifies the arch metadata of a pip package linux_x86_64=>manylinux1_x86_64
    :param name:
    :param version:
    :return:
    """
    # Get python version as XY (27, 35, 36, etc)
    python_version = str(sys.version_info.major) + str(sys.version_info.minor)
    name_version = '{}-{}'.format(name.replace('-', '_'), version)

    # linux wheel package
    dist_zip = '{0}-cp{1}-cp{1}m-linux_x86_64.whl'.format(name_version,
                                                          python_version)
    dist_zip_path = os.path.join('dist', dist_zip)
    if not os.path.exists(dist_zip_path):
        print('Wheel not found: {}'.format(dist_zip_path))
        return

    unzip_dir = 'dist/unzip'
    os.makedirs(unzip_dir, exist_ok=True)
    with zipfile.ZipFile(dist_zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)

    wheel_file = '{}.dist-info/WHEEL'.format(name_version)
    new_wheel_str = ''
    with open(os.path.join(unzip_dir, wheel_file)) as f:
        for line in f.readlines():
            if line.startswith('Tag'):
                # Replace the linux tag
                new_wheel_str += line.replace('linux', 'manylinux1')
            else:
                new_wheel_str += line

    # compute hash & size of the new WHEEL file
    # Follows https://www.python.org/dev/peps/pep-0376/#record
    m = sha256()
    m.update(new_wheel_str.encode('utf-8'))
    hash = urlsafe_b64encode(m.digest()).decode('utf-8')
    hash = hash.replace('=', '')

    with open(os.path.join(unzip_dir, wheel_file), 'w') as f:
        f.write(new_wheel_str)
    statinfo = os.stat(os.path.join(unzip_dir, wheel_file))
    byte_size = statinfo.st_size

    record_file = os.path.join(unzip_dir,
                               '{}.dist-info/RECORD'.format(name_version))
    new_record_str = ''
    with open(record_file) as f:
        for line in f.readlines():
            if line.startswith(wheel_file):
                # Update the record for the WHEEL file
                new_record_str += '{},sha256={},{}'.format(wheel_file, hash,
                                                           str(byte_size))
                new_record_str += os.linesep
            else:
                new_record_str += line

    with open(record_file, 'w') as f:
        f.write(new_record_str)

    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.join(root, file).replace(path, ''))

    new_zip_name = dist_zip.replace('linux', 'manylinux1')
    print('Generating new zip {}...'.format(new_zip_name))
    zipf = zipfile.ZipFile(os.path.join('dist', new_zip_name),
                           'w', zipfile.ZIP_DEFLATED)
    zipdir(unzip_dir, zipf)
    zipf.close()

    shutil.rmtree(unzip_dir, ignore_errors=True)
    os.remove(dist_zip_path)


convert_to_manylinux(name, version)
