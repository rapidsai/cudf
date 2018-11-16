# libcudf: The CUDA DataFrame Library

libcudf is a C/C++ CUDA library for implementing standard dataframe operations.

## Tested Development Platform

Currently, building `libcudf` from source is only tested on Ubuntu 16.04 Linux
(64-bit).

Target build system:

* `gcc`     version 5.4
* `nvcc`    version 9.2
* `cmake`   version 3.12

`libcudf` has been tested with `nvcc` versions 9.2 and 10.0.

You can obtain CUDA from 
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### Dependencies

`libcudf` requires the following:

* Apache Arrow          `0.10.0`
* Google Test           `1.8.0`
* Boost C++ Library     `1.58`
* CMake                 `3.12`

#### Optional Python Dependencies

Python version `3.6` is recommended.

* Cython                `0.29`
* PyArrow               `0.10.0`
* PyTest                `4.0.0`

## Conda Environment Configuration

We recommend setting up a conda environment for the dependencies using the 
provided configuration file.

```bash
# create the conda environment (assuming `pwd` is base `cudf` directory)
$ conda env create --name cudf_dev --file conda_environments/dev_py35.yml
# activate the environment
$ source activate cudf_dev
# when not using default arrow version 0.10.0, run
$ conda install pyarrow=$ARROW_VERSION -c conda-forge
```

This installs the required `cmake`, `nvstrings`, `pyarrow` and other 
dependencies into the `cudf_dev` conda environment and activates it.

More information: the python cffi wrapper code requires `cffi` and `pytest`.
The testing code requires `numba` and `pandas`. IPC testing requires 
`distributed`. All of these are installed from the previous commands.

The environment can be updated from `conda_environments/dev_py35.yml` as
development includes/changes the depedencies. To do so, run:

```bash
# Update the conda environment (assuming `pwd` is base `cudf` directory)
conda env update --name cudf_dev --file conda_environments/dev_py35.yml
```
Note that `dev_py35.yml` uses pyarrow 0.10.0. Reinstall pyarrow if 
needed using `conda install pyarrow=$ARROW_VERSION -c conda-forge`.

### Configure and Build

Use CMake to configure the build files:

```bash
$ cd /path/to/cudf/cpp                              # navigate to C/C++ CUDA source root directory
$ mkdir build                                       # make a build directory
$ cd build                                          # enter the build directory
$ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path     # configure cmake ... use $CONDA_PREFIX if you're using Anaconda
$ make -j                                           # compile the libraries librmm.so, libcudf.so ... '-j' will start a parallel job using the number of physical cores available on your system
$ make install                                      # install the libraries librmm.so, libcudf.so to '/install/path'
```

To run tests, call:

```bash
$ make test
```

The correct output will be of the following form:

```bash
$ make test
Running tests...
Test project /home/nfs/majones/workspace/github/rapids/cudf/cpp/build
      Start  1: COLUMN_TEST
 1/14 Test  #1: COLUMN_TEST ......................   Passed    1.20 sec
      Start  2: CUDF_INTERNAL_TEST
 2/14 Test  #2: CUDF_INTERNAL_TEST ...............   Passed    0.01 sec
 
     ...

      Start 14: RMM_TEST
14/14 Test #14: RMM_TEST .........................   Passed    1.15 sec

100% tests passed, 0 tests failed out of 14

Total Test time (real) =  74.77 sec
```

#### Optional Python Bindings

```bash
make python_cffi                                    # build CFFI bindings for librmm.so, libcudf.so
make install_python                                 # install python bindings into site-packages
```
#### Optional Python Tests

```bash
$ cd src/build/python
$ py.test -v
```
