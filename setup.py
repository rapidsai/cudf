from setuptools import setup

setup(name='libgdf',
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["libgdf_build.py:ffibuilder"],
      install_requires=["cffi>=1.0.0"],
      )


