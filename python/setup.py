from setuptools import setup

setup(name='libgdf_cffi',
      version="0.1.0a1",
      packages=["libgdf_cffi"],
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["libgdf_cffi/libgdf_build.py:ffibuilder"],
      install_requires=["cffi>=1.0.0"],
      )


