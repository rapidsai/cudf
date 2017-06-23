# libgdf: GPU Dataframes

libgdf is a C library for implementing common functionality for a GPU Data Frame.  For more project details, see [the wiki](https://github.com/gpuopenanalytics/libgdf/wiki/Home).

# Development Setup

The following instructions are tested on Linux and OSX systems.

## Get dependencies

It is recommended to setup a conda environment for the dependencies.

```bash
# create the conda environment (assuming in build directory)
$ conda env create --name libgdf_dev --file ../conda_environments/dev_py35.yml
# activate the environment
$ source activate libgdf_dev
```

This installs the required `cmake`, `flatbuffers` into the `libgdf_dev` conda
environment and activates it.

For additional information, the python cffi wrapper code requires `cffi` and
`pytest`.  The testing code requires `numba` and `cudatoolkit` as an
additional dependency.  All these are installed from the previous commands.

The environment can be updated from `../conda_environments/dev_py35.yml` as
development includes/changes the depedencies.  To do so, run:

```bash
$ conda env update --name libgdf_dev --file ../conda_environments/dev_py35.yml
```

## Configure and build

This project uses cmake for building the C/C++ library.  To configure cmake,
run:

```bash
mkdir build   # create build directory for out-of-source build
cd build      # enter the build directory
cmake ..      # configure cmake
```

To build the C/C++ code, run `make`.  This should produce a shared library
named `libgdf.so` or `libgdf.dylib`.


## Link python files into the build directory

To make development and testing more seamless, the python files and tests
can be symlinked into the build directory by running `make copy_python`.
With that, any changes to the python files are reflected in the build
directory.  To rebuild the libgdf, run `make` again.

## Run tests

Currently, all tests are written in python with `py.test`.  A make target is
available to trigger the test execution.  In the build directory (and with the
conda environment activated), run below to exceute test:

```bash
make pytest   # this auto trigger target "copy_python"
```

