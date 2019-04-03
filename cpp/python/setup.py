from setuptools import setup

setup(name='libcudf_cffi',
      version="0.6.0",
      packages=["libcudf_cffi"],
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["libcudf_cffi/libcudf_build.py:ffibuilder"],
      install_requires=["cffi>=1.0.0"],
      )
