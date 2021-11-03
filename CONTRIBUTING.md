# Contributing to cuDF

Contributions to cuDF fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/rapidsai/cudf/issues/new/choose)
    describing in detail the problem or new feature. The RAPIDS team evaluates
    and triages issues, and schedules them for a release. If you believe the
    issue needs priority attention, please comment on the issue to notify the
    team.
2. To propose and implement a new Feature, please file a new feature request
    [issue](https://github.com/rapidsai/cudf/issues/new/choose). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and
    implement it, using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please
    Follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, please ask in a comment.

As contributors and maintainers to this project,
you are expected to abide by cuDF's code of conduct.
More information can be found at: [Contributor Code of Conduct](https://docs.rapids.ai/resources/conduct/).

## Code contributions

### Your first issue

1. Follow the guide at the bottom of this page for [Setting Up Your Build Environment](#setting-up-your-build-environment).
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/cudf/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/cudf/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels.
3. Comment on the issue stating that you are going to work on it.
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/cudf/compare).
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed.
7. Wait for other developers to review your code and update code as needed.
8. Once reviewed and approved, a RAPIDS developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues for our next release in our [project boards](https://github.com/rapidsai/cudf/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue to which you are comfortable
contributing. Start with _Step 3_ above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Setting Up Your Build Environment

The following instructions are for developers and contributors to cuDF OSS development. These instructions are tested on Linux Ubuntu 16.04 & 18.04. Use these instructions to build cuDF from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.



### General requirements

Compilers:

* `gcc`     version 9.3+
* `nvcc`    version 11.5+
* `cmake`   version 3.20.1+

CUDA/GPU:

* CUDA 11.5+
* NVIDIA driver 450.80.02+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Create the build Environment

- Clone the repository and submodules
```bash
CUDF_HOME=$(pwd)/cudf
git clone https://github.com/rapidsai/cudf.git $CUDF_HOME
cd $CUDF_HOME
git submodule update --init --remote --recursive
```
- Create the conda development environment `cudf_dev`:
```bash
# create the conda environment (assuming in base `cudf` directory)
# note: RAPIDS currently doesn't support `channel_priority: strict`; use `channel_priority: flexible` instead
conda env create --name cudf_dev --file conda/environments/cudf_dev_cuda11.0.yml
# activate the environment
conda activate cudf_dev
```
- For other CUDA versions, check the corresponding cudf_dev_cuda*.yml file in conda/environments

### Build cuDF from source

- A `build.sh` script is provided in `$CUDF_HOME`. Running the script with no additional arguments will install the `libcudf`, `cudf` and `dask_cudf` libraries. By default, the libraries are installed to the `$CONDA_PREFIX` directory. To install into a different location, set the location in `$INSTALL_PREFIX`. Finally, note that the script depends on the `nvcc` executable being on your path, or defined in `$CUDACXX`.
```bash
cd $CUDF_HOME

# Choose one of the following commands, depending on whether
# you want to build and install the libcudf C++ library only, 
# or include the cudf and/or dask_cudf Python libraries:

./build.sh  # libcudf, cudf and dask_cudf
./build.sh libcudf  # libcudf only
./build.sh libcudf cudf  # libcudf and cudf only             
```
- Other libraries like `cudf-kafka` and `custreamz` can be installed with this script. For the complete list of libraries as well as details about the script usage, run the `help` command:
```bash
./build.sh --help            
```

### Build, install and test cuDF libraries for contributors

The general workflow is provided below. Please, also see the last section about [code formatting](###code-formatting).

#### `libcudf` (C++)

If you're only interested in building the library (and not the unit tests):
 
```bash
cd $CUDF_HOME
./build.sh libcudf
```
If, in addition, you want to build tests:

```bash
./build.sh libcudf tests
```
To run the tests:

```bash
make test                                      
```

#### `cudf` (Python)

- First, build the `libcudf` C++ library following the steps above

- To build and install in edit/develop `cudf` python package:
```bash
cd $CUDF_HOME/python/cudf
python setup.py build_ext --inplace
python setup.py develop
```

- To run `cudf` tests :
```bash
cd $CUDF_HOME/python
py.test -v cudf/cudf/tests
```

#### `dask-cudf` (Python)

- First, build the `libcudf` C++ and `cudf` Python libraries following the steps above

- To install in edit/develop mode the `dask-cudf` python package:
```bash
cd $CUDF_HOME/python/dask_cudf
python setup.py build_ext --inplace
python setup.py develop
```

- To run `dask_cudf` tests :
```bash
cd $CUDF_HOME/python
py.test -v dask_cudf
```

#### `libcudf_kafka` (C++)

If you're only interested in building the library (and not the unit tests):
 
```bash
cd $CUDF_HOME
./build.sh libcudf_kafka
```
If, in addition, you want to build tests:

```bash
./build.sh libcudf_kafka tests
```
To run the tests:

```bash
make test                                      
```

#### `cudf-kafka` (Python)

- First, build the `libcudf` and `libcudf_kafka` following the steps above

- To install in edit/develop mode the `cudf-kafka` python package:
```bash
cd $CUDF_HOME/python/cudf_kafka
python setup.py build_ext --inplace
python setup.py develop
```

#### `custreamz` (Python)

- First, build `libcudf`, `libcudf_kafka`, and `cudf_kafka` following the steps above

- To install in edit/develop mode the `custreamz` python package:
```bash
cd $CUDF_HOME/python/custreamz
python setup.py build_ext --inplace
python setup.py develop
```

- To run `custreamz` tests :
```bash
cd $CUDF_HOME/python
py.test -v custreamz
```

#### `cudf` (Java):

- First, build the `libcudf` C++ library following the steps above

- Then, refer to [Java README](https://github.com/rapidsai/cudf/blob/branch-21.10/java/README.md)


Done! You are ready to develop for the cuDF OSS project. But please go to [code formatting](###code-formatting) to ensure that you contributing code follows the expected format.

## Debugging cuDF

### Building Debug mode from source

Follow the [above instructions](####build-cudf-from-source) to build from source and add `-g` to the `./build.sh` command.

For example:
```bash
./build.sh libcudf -g
```

This builds `libcudf` in Debug mode which enables some `assert` safety checks and includes symbols in the library for debugging.

All other steps for installing `libcudf` into your environment are the same.

### Debugging with `cuda-gdb` and `cuda-memcheck`

When you have a debug build of `libcudf` installed, debugging with the `cuda-gdb` and `cuda-memcheck` is easy.

If you are debugging a Python script, simply run the following:

```bash
cuda-gdb -ex r --args python <program_name>.py <program_arguments>
```

```bash
cuda-memcheck python <program_name>.py <program_arguments>
```

### Device debug symbols

The device debug symbols are not automatically added with the cmake `Debug`
build type because it causes a runtime delay of several minutes when loading
the libcudf.so library.

Therefore, it is recommended to add device debug symbols only to specific files by
setting the `-G` compile option locally in your `cpp/CMakeLists.txt` for that file.
Here is an example of adding the `-G` option to the compile command for
`src/copying/copy.cu` source file:

```
set_source_files_properties(src/copying/copy.cu PROPERTIES COMPILE_OPTIONS "-G")
```

This will add the device debug symbols for this object file in libcudf.so.
You can then use `cuda-dbg` to debug into the kernels in that source file.

### Building and Testing on a gpuCI image locally

Before submitting a pull request, you can do a local build and test on your machine that mimics our gpuCI environment using the `ci/local/build.sh` script.
For detailed information on usage of this script, see [here](ci/local/README.md).


## Automated Build in Docker Container

A Dockerfile is provided with a preconfigured conda environment for building and installing cuDF from source based off of the main branch.

### Prerequisites

* Install [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) for Docker + GPU support
* Verify NVIDIA driver is `450.80.02` or higher
* Ensure CUDA 11.0+ is installed

### Usage

From cudf project root run the following, to build with defaults:
```bash
docker build --tag cudf .
```
After the container is built run the container:
```bash
docker run --runtime=nvidia -it cudf bash
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
| `CUDA_VERSION` | 11.0 | 11.2.2 | set CUDA version |
| `LINUX_VERSION` | ubuntu18.04 | ubuntu20.04 | set Ubuntu version |
| `CC` & `CXX` | 9 | 10 | set gcc/g++ version |
| `CUDF_REPO` | This repo | Forks of cuDF | set git URL to use for `git clone` |
| `CUDF_BRANCH` | main | Any branch name | set git branch to checkout of `CUDF_REPO` |
| `NUMBA_VERSION` | newest | >=0.40.0 | set numba version |
| `NUMPY_VERSION` | newest | >=1.14.3 | set numpy version |
| `PANDAS_VERSION` | newest | >=0.23.4 | set pandas version |
| `PYARROW_VERSION` | 1.0.1 | Not supported | set pyarrow version |
| `CMAKE_VERSION` | newest | >=3.18 | set cmake version |
| `CYTHON_VERSION` | 0.29 | Not supported | set Cython version |
| `PYTHON_VERSION` | 3.7 | 3.8 | set python version |


### Code Formatting


#### Python

cuDF uses [Black](https://black.readthedocs.io/en/stable/),
[isort](https://readthedocs.org/projects/isort/), and
[flake8](http://flake8.pycqa.org/en/latest/) to ensure a consistent code format
throughout the project. They have been installed during the `cudf_dev` environment creation.

These tools are used to auto-format the Python code, as well as check the Cython
code in the repository. Additionally, there is a CI check in place to enforce
that committed code follows our standards. You can use the tools to
automatically format your python code by running:

```bash
isort --atomic python/**/*.py
black python
```

and then check the syntax of your Python and Cython code by running:

```bash
flake8 python
flake8 --config=python/.flake8.cython
```

Additionally, many editors have plugins that will apply `isort` and `Black` as
you edit files, as well as use `flake8` to report any style / syntax issues.

#### C++/CUDA

cuDF uses [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html)

In order to format the C++/CUDA files, navigate to the root (`cudf`) directory and run:
```
python3 ./cpp/scripts/run-clang-format.py -inplace
```

Additionally, many editors have plugins or extensions that you can set up to automatically run `clang-format` either manually or on file save.

#### Pre-commit hooks

Optionally, you may wish to setup [pre-commit hooks](https://pre-commit.com/)
to automatically run `isort`, `Black`, `flake8` and `clang-format` when you make a git commit.
This can be done by installing `pre-commit` via `conda` or `pip`:

```bash
conda install -c conda-forge pre_commit
```

```bash
pip install pre-commit
```

and then running:

```bash
pre-commit install
```

from the root of the cuDF repository. Now `isort`, `Black`, `flake8` and `clang-format` will be
run each time you commit changes.

---

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
Portions adopted from https://github.com/dask/dask/blob/master/docs/source/develop.rst
