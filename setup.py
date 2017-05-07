from setuptools import setup

import versioneer


packages = ['pygdf',
            'pygdf.sorting',
            'pygdf.tests',
            ]

install_requires = [
    'numba',
]

setup(name='pygdf',
      description="GPU Dataframe",
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
      author="Continuum Analytics, Inc.",
      packages=packages,
      install_requires=install_requires,
      license="BSD",
      cmdclass=versioneer.get_cmdclass(),
      )
