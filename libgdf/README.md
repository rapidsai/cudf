# libgdf: GPU DataFrames

libgdf is a C library for implementing common functionality for a GPU DataFrame.

## Development Setup

The following instructions are tested on Linux and OSX systems.

Compiler requirement:

* `g++` 5.4
* `cmake` 3.12+

CUDA requirement:

* CUDA 9.2+

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### Get dependencies

Since `cmake` will download and build Apache Arrow (version 0.7.1 or
0.8+) you may need to install Boost C++ (version 1.58) before running
`cmake`:

```bash
# Install Boost C++ 1.58 for Ubuntu 16.04
$ sudo apt-get install libboost-all-dev
```

or

```bash
# Install Boost C++ 1.58 for Conda (you will need a Python 3.3 environment)
$ conda install -c omnia boost=1.58.0=py33_0
```

Libgdf supports Apache Arrow versions 0.7.1 and 0.8+ (0.10.0 is
default) that use different metadata versions in IPC. So, it is
important to specify which Apache arrow version will be used during
building libgdf.  To select required Apache Arrow version, define the
following environment variables (using Arrow version 0.10.0 as an
example):
```bash
$ export ARROW_VERSION=0.10.0
$ export PARQUET_ARROW_VERSION=apache-arrow-$ARROW_VERSION
```
where the latter is used by libgdf cmake configuration files. Note
that when using libgdf, defining the above environment variables is
not necessary.

You can install Boost C++ 1.58 from sources as well: https://www.boost.org/doc/libs/1_58_0/more/getting_started/unix-variants.html

To run the python tests it is recommended to setup a conda environment for 
the dependencies.

```bash
# create the conda environment (assuming in build directory)
$ conda env create --name libgdf_dev --file conda_environments/dev_py35.yml
# activate the environment
$ source activate libgdf_dev
# when not using default arrow version 0.10.0, run
$ conda install pyarrow=$ARROW_VERSION -c conda-forge
```

This installs the required `cmake` and `pyarrow` into the `libgdf_dev` conda
environment and activates it.

For additional information, the python cffi wrapper code requires `cffi` and
`pytest`.  The testing code requires `numba` and `cudatoolkit` as an
additional dependency.  All these are installed from the previous commands.

The environment can be updated from `conda_environments/dev_py35.yml` as
development includes/changes the depedencies.  To do so, run:

```bash
conda env update --name libgdf_dev --file conda_environments/dev_py35.yml
```
Note that `dev_py35.yml` uses the latest version of pyarrow.
Reinstall pyarrow if needed using `conda install
pyarrow=$ARROW_VERSION -c conda-forge`.

### Configure and build

This project uses cmake for building the C/C++ library. To configure cmake,
run:

```bash
$ mkdir build   # create build directory for out-of-source build
$ cd build      # enter the build directory
$ cmake ..      # configure cmake (will download and build Apache Arrow and Google Test)
```

If installing libgdf to conda environment is desired, then replace the last command with
```bash
$ cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
```

To build the C/C++ code, run `make`.  This should produce a shared library
named `libgdf.so` or `libgdf.dylib`.

If you run into compile errors about missing header files:

```bash
cub/device/device_segmented_radix_sort.cuh: No such file or directory
```

See the note about submodules in the Get dependencies section above.

### Link python files into the build directory

To make development and testing more seamless, the python files and tests
can be symlinked into the build directory by running `make copy_python`.
With that, any changes to the python files are reflected in the build
directory.  To rebuild the libgdf, run `make` again.

### Run tests

Currently, all tests are written in python with `py.test`.  A make target is
available to trigger the test execution.  In the build directory (and with the
conda environment activated), run below to exceute test:

```bash
$ make pytest   # this auto trigger target "copy_python"
```
