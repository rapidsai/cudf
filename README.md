# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuDF - GPU DataFrames</div>

[![Build Status](http://18.191.94.64/buildStatus/icon?job=cudf-master)](http://18.191.94.64/job/cudf-master/)&nbsp;&nbsp;[![Documentation Status](https://readthedocs.org/projects/cudf/badge/?version=latest)](https://cudf.readthedocs.io/en/latest/)

The [RAPIDS](https://rapids.ai) cuDF library is a GPU DataFrame manipulation library based on Apache Arrow that accelerates loading, filtering, and manipulation of data for model training data preparation. The RAPIDS GPU DataFrame provides a pandas-like API that will be familiar to data scientists, so they can now build GPU-accelerated workflows more easily.

## Quick Start

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize cuDF.

## Install cuDF

### Conda

It is easy to install cuDF using conda. You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html) or get the full installation with [Anaconda](https://www.anaconda.com/download).

Install and update cuDF using the conda command:

```bash
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cudf
```

Note: This conda installation only applies to Linux and Python versions 3.6/3.7.

### Pip

It is easy to install cuDF using pip. You must specify the CUDA version to ensure you install the right package.

```bash
pip install cudf-cuda92 # CUDA 9.2
pip install cudf-cuda100 # CUDA 10.0
```

## Development Setup

The following instructions are for developers and contributors to cuDF OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuDF from source and contribute to its development.  Other operatings systems may be compatible, but are not currently tested.

### Get libcudf Dependencies

Compiler requirements:

* `gcc`     version 5.4+
* `nvcc`    version 9.2
* `cmake`   version 3.12

CUDA/GPU requirements:

* CUDA 9.2+
* NVIDIA driver 396.44+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Since `cmake` will download and build Apache Arrow you may need to install Boost C++ (version 1.58+) before running
`cmake`:

```bash
# Install Boost C++ for Ubuntu 16.04/18.04
$ sudo apt-get install libboost-all-dev
```

or

```bash
# Install Boost C++ for Conda
$ conda install -c conda-forge boost
```

## Script to build cuDF from source

### Build from Source

To install cuDF from source, ensure the dependencies are met and follow the steps below:

- Clone the repository and submodules
```bash
CUDF_HOME=$(pwd)/cudf
git clone --recurse-submodules https://github.com/rapidsai/cudf.git $CUDF_HOME
cd CUDF_HOME
```
- Create the conda development environment `cudf_dev`
```bash
# create the conda environment (assuming in base `cudf` directory)
conda env create --name cudf_dev --file conda/environments/cudf_dev.yml
# activate the environment
source activate cudf_dev
```

- Build and install `libcudf`. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.
```bash
$ cd $CUDF_HOME/cpp                                                       # navigate to C/C++ CUDA source root directory
$ mkdir build                                                             # make a build directory
$ cd build                                                                # enter the build directory

# CMake options:
# -DCMAKE_INSTALL_PREFIX set to the install path for your libraries or $CONDA_PREFIX if you're using Anaconda, i.e. -DCMAKE_INSTALL_PREFIX=/install/path or -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
# -DCMAKE_CXX11_ABI set to ON or OFF depending on the ABI version you want, defaults to OFF. When turned ON, ABI compability for C++11 is used. When OFF, pre-C++11 ABI compability is used.
$ cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=OFF     # configure cmake ...

$ make -j                                                                 # compile the libraries librmm.so, libcudf.so ... '-j' will start a parallel job using the number of physical cores available on your system
$ make install                                                            # install the libraries librmm.so, libcudf.so to the CMAKE_INSTALL_PREFIX
```

- To run tests (Optional):
```bash
$ make test
```

- Build, install, and test cffi bindings:
```bash
$ make python_cffi                                  # build CFFI bindings for librmm.so, libcudf.so
$ make install_python                               # build & install CFFI python bindings. Depends on cffi package from PyPi or Conda
$ cd python && py.test -v                           # optional, run python tests on low-level python bindings
```

- 4. Build the `cudf` python package, in the `python` folder:
```bash
$ cd $CUDF_HOME/python
$ python setup.py build_ext --inplace
```

- You will also need the following environment variables, including `$CUDA_HOME`.
```bash
NUMBAPRO_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so
NUMBAPRO_LIBDEVICE=$CUDA_HOME/nvvm/libdevice
```

- To run Python tests (Optional):
```bash
$ py.test -v                                        # run python tests on cudf python bindings
```

- Finally, install the Python package to your Python path:
```bash
$ python setup.py install                           # install cudf python bindings
```

Done! You are ready to develop for the cuDF OSS project.

## Debugging cuDF

### Building Debug mode from source

Follow the [above instructions](#build-from-source) to build from source and add `-DCMAKE_BUILD_TYPE=Debug` to the `cmake` step. 

For example:
```bash
$ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path -DCMAKE_BUILD_TYPE=Debug     # configure cmake ... use -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX if you're using Anaconda
```

This builds `libcudf` in Debug mode which enables some `assert` safety checks and includes symbols in the library for debugging.

All other steps for installing `libcudf` into your environment are the same.

### Debugging with `cuda-gdb` and `cuda-memcheck`

When you have a debug build of `libcudf` installed, debugging with the `cuda-gdb` and `cuda-memcheck` is easy.

If you are debugging a Python script, simply run the following:

#### `cuda-gdb`

```bash
cuda-gdb -ex r --args python <program_name>.py <program_arguments>
```

#### `cuda-memcheck`

```bash
cuda-memcheck python <program_name>.py <program_arguments>
```


## Automated Build in Docker Container

A Dockerfile is provided with a preconfigured conda environment for building and installing cuDF from source based off of the master branch.

### Prerequisites

* Install [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) for Docker + GPU support
* Verify NVIDIA driver is `396.44` or higher
* Ensure CUDA 9.2+ is installed

### Usage

From cudf project root run the following, to build with defaults:
```bash
$ docker build --tag cudf .
```
After the container is built run the container:
```bash
$ docker run --runtime=nvidia -it cudf bash
```
Activate the conda environment `cudf` to use the newly built cuDF and libcudf libraries:
```
root@3f689ba9c842:/# source activate cudf
(cudf) root@3f689ba9c842:/# python -c "import cudf"
(cudf) root@3f689ba9c842:/#
```

### Customizing the Build

Several build arguments are available to customize the build process of the
container. These are specified by using the Docker [build-arg](https://docs.docker.com/engine/reference/commandline/build/#set-build-time-variables---build-arg)
flag. Below is a list of the available arguments and their purpose:

| Build Argument | Default Value | Other Value(s) | Purpose |
| --- | --- | --- | --- |
| `CUDA_VERSION` | 9.2 | 10.0 | set CUDA version |
| `LINUX_VERSION` | ubuntu16.04 | ubuntu18.04 | set Ubuntu version |
| `CC` & `CXX` | 5 | 7 | set gcc/g++ version; **NOTE:** gcc7 requires Ubuntu 18.04 |
| `CUDF_REPO` | This repo | Forks of cuDF | set git URL to use for `git clone` |
| `CUDF_BRANCH` | master | Any branch name | set git branch to checkout of `CUDF_REPO` |
| `NUMBA_VERSION` | 0.40.0 | Not supported | set numba version |
| `NUMPY_VERSION` | 1.14.3 | Not supported | set numpy version |
| `PANDAS_VERSION` | 0.24.3 | Not supported | set pandas version |
| `PYARROW_VERSION` | 0.11.1 | Not supported | set pyarrow version |
| `PYTHON_VERSION` | 3.6 | 3.7 | set python version |

---

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>

### Apache Arrow on GPU

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.
