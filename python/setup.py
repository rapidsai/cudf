from setuptools import setup

setup(name='libgdf_cffi',
      packages=["libgdf_cffi"],
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["libgdf_cffi/libgdf_build.py:ffibuilder"],
      install_requires=["cffi>=1.0.0"],
      )


