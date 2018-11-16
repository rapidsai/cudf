# libcudf: The CUDA DataFrame Library

libcudf is a C/C++ CUDA library for implementing standard dataframe operations.

## Development Setup

Currently, only Ubuntu 16.04 is supported for building `libcudf` from source.

Target build system:

* `gcc`     version 5.4
* `nvcc`    version 9.2
* `cmake`   version 3.12

`libcudf` has been tested with `nvcc` version 9.2, 10.0. More detailed information on the supported version follows:

```bash
$ gcc --version
gcc (Ubuntu 5.4.0-6ubuntu1~16.04.10) 5.4.0 20160609
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:07:04_CDT_2018
Cuda compilation tools, release 9.2, V9.2.148
```

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### Dependencies

`libcudf` requires the following:

* Apache Arrow          `0.10.0`
* Google Test           `1.8.0`
* Boost C++ Library     `1.58`

#### Optional Python Dependencies

Python version `3.6` is recommended.

* Cython                `0.29`
* PyArrow               `0.10.0`
* PyTest                `4.0.0`

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
      Start  3: FILTER_TEST
 3/14 Test  #3: FILTER_TEST ......................   Passed   11.85 sec
      Start  4: GROUPBY_TEST
 4/14 Test  #4: GROUPBY_TEST .....................   Passed    2.74 sec
      Start  5: JOIN_TEST
 5/14 Test  #5: JOIN_TEST ........................   Passed   39.39 sec
      Start  6: SQLS_TEST
 6/14 Test  #6: SQLS_TEST ........................   Passed    1.20 sec
      Start  7: BITMASK_TEST
 7/14 Test  #7: BITMASK_TEST .....................   Passed    2.23 sec
      Start  8: DATETIME_TEST
 8/14 Test  #8: DATETIME_TEST ....................   Passed    1.17 sec
      Start  9: HASHING_TEST
 9/14 Test  #9: HASHING_TEST .....................   Passed    5.31 sec
      Start 10: HASH_MAP_TEST
10/14 Test #10: HASH_MAP_TEST ....................   Passed    6.03 sec
      Start 11: QUANTILES_TEST
11/14 Test #11: QUANTILES_TEST ...................   Passed    1.19 sec
      Start 12: UNARY_TEST
12/14 Test #12: UNARY_TEST .......................   Passed    1.28 sec
      Start 13: CSV_TEST
13/14 Test #13: CSV_TEST .........................   Passed    0.01 sec
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



