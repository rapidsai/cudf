from setuptools import setup

import versioneer


packages = ['cudf',
            'cudf.tests',
            ]

install_requires = [
    'numba',
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
      packages=packages,
      package_data={
        'cudf.tests': ['data/*.pickle'],
      },
      install_requires=install_requires,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      )
