from setuptools import setup

setup(name='libgdf_cffi',
      version="0.2.0",
      packages=["libgdf_cffi"],
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["libgdf_cffi/libgdf_build.py:ffibuilder"],
      install_requires=["cffi>=1.0.0"],
      )

setup(name='librmm_cffi',
      version="0.2.0",
      packages=["librmm_cffi"],
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["librmm_cffi/librmm_build.py:ffibuilder"],
      install_requires=["cffi>=1.0.0"],
      )

