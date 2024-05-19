# Contributing to cuDF

Contributions to cuDF fall into the following categories:

1. To report a bug, request a new feature, or report a problem with documentation, please file an
   [issue](https://github.com/rapidsai/cudf/issues/new/choose) describing the problem or new feature
   in detail. The RAPIDS team evaluates and triages issues, and schedules them for a release. If you
   believe the issue needs priority attention, please comment on the issue to notify the team.
2. To propose and implement a new feature, please file a new feature request
   [issue](https://github.com/rapidsai/cudf/issues/new/choose). Describe the intended feature and
   discuss the design and implementation with the team and community. Once the team agrees that the
   plan looks good, go ahead and implement it, using the [code contributions](#code-contributions)
   guide below.
3. To implement a feature or bug fix for an existing issue, please follow the [code
   contributions](#code-contributions) guide below. If you need more context on a particular issue,
   please ask in a comment.

As contributors and maintainers to this project, you are expected to abide by cuDF's code of
conduct. More information can be found at:
[Contributor Code of Conduct](https://docs.rapids.ai/resources/conduct/).

## Code contributions

### Your first issue

1. Follow the guide at the bottom of this page for
   [Setting up your build environment](#setting-up-your-build-environment).
2. Find an issue to work on. The best way is to look for the
   [good first issue](https://github.com/rapidsai/cudf/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
   or [help wanted](https://github.com/rapidsai/cudf/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
   labels.
3. Comment on the issue stating that you are going to work on it.
4. Create a fork of the cudf repository and check out a branch with a name that
   describes your planned work. For example, `fix-documentation`.
5. Write code to address the issue or implement the feature.
6. Add unit tests and unit benchmarks.
7. [Create your pull request](https://github.com/rapidsai/cudf/compare). To run continuous integration (CI) tests without requesting review, open a draft pull request.
8. Verify that CI passes all [status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks).
   Fix if needed.
9. Wait for other developers to review your code and update code as needed.
10. Once reviewed and approved, a RAPIDS developer will merge your pull request.

If you are unsure about anything, don't hesitate to comment on issues and ask for clarification!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you can look at the
prioritized issues for our next release in our
[project boards](https://github.com/rapidsai/cudf/projects).

**Note:** Always look at the release board that is
[currently under development](https://docs.rapids.ai/maintainers) for issues to work on. This is
where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue to which you are comfortable contributing. Start
with _Step 3_ above, commenting on the issue to let others know you are working on it. If you have
any questions related to the implementation of the issue, ask them in the issue instead of the PR.

## Setting up your build environment

The following instructions are for developers and contributors to cuDF development. These
instructions are tested on Ubuntu Linux LTS releases. Use these instructions to build cuDF from
source and contribute to its development. Other operating systems may be compatible, but are not
currently tested.

Building cudf with the provided conda environment is recommended for users who wish to enable all
library features. The following instructions are for building with a conda environment. Dependencies
for a minimal build of libcudf without using conda are also listed below.

### General requirements

Compilers:

* `gcc` version 9.3+
* `nvcc` version 11.5+
* `cmake` version 3.26.4+

CUDA/GPU:

* CUDA 11.5+
* NVIDIA driver 450.80.02+
* Volta architecture or better (Compute Capability >=7.0)

You can obtain CUDA from
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

### Create the build environment

- Clone the repository:

```bash
CUDF_HOME=$(pwd)/cudf
git clone https://github.com/rapidsai/cudf.git $CUDF_HOME
cd $CUDF_HOME
```

#### Building with a conda environment

**Note:** Using a conda environment is the easiest way to satisfy the library's dependencies.
Instructions for a minimal build environment without conda are included below.

- Create the conda development environment:

```bash
# create the conda environment (assuming in base `cudf` directory)
# note: RAPIDS currently doesn't support `channel_priority: strict`;
# use `channel_priority: flexible` instead
conda env create --name cudf_dev --file conda/environments/all_cuda-122_arch-x86_64.yaml
# activate the environment
conda activate cudf_dev
```

- **Note**: the conda environment files are updated frequently, so the
  development environment may also need to be updated if dependency versions or
  pinnings are changed.

#### Building without a conda environment

- libcudf has the following minimal dependencies (in addition to those listed in the [General
  requirements](#general-requirements)). The packages listed below use Ubuntu package names:

  - `build-essential`
  - `libssl-dev`
  - `libz-dev`
  - `libpython3-dev` (required if building cudf)

### Build cuDF from source

- A `build.sh` script is provided in `$CUDF_HOME`. Running the script with no additional arguments
  will install the `libcudf`, `cudf` and `dask_cudf` libraries. By default, the libraries are
  installed to the `$CONDA_PREFIX` directory. To install into a different location, set the location
  in `$INSTALL_PREFIX`. Finally, note that the script depends on the `nvcc` executable being on your
  path, or defined in `$CUDACXX`.

```bash
cd $CUDF_HOME

# Choose one of the following commands, depending on whether
# you want to build and install the libcudf C++ library only,
# or include the cudf and/or dask_cudf Python libraries:

./build.sh  # libcudf, cudf and dask_cudf
./build.sh libcudf  # libcudf only
./build.sh libcudf cudf  # libcudf and cudf only
```

- Other libraries like `cudf-kafka` and `custreamz` can be installed with this script. For the
  complete list of libraries as well as details about the script usage, run the `help` command:

```bash
./build.sh --help
```

#### Building for development

To build Python packages for development purposes, add the `--pydevelop` flag.
To build C++ tests, you can also request that build.sh build the `test` target.
To build all libraries and tests, with Python packages in development mode, simply run

```bash
./build.sh --pydevelop libcudf libcudf_kafka cudf dask_cudf cudf_kafka custreamz
```

- **Note**: if Cython files (`*.pyx` or `*.pxd`) have changed, the Python build must be rerun.

To run the C++ tests, run

```bash
ctest --test-dir ${CUDF_HOME}/cpp/build  # libcudf
ctest --test-dir ${CUDF_HOME}/cpp/libcudf_kafka/build  # libcudf_kafka
```

To run python tests, run

- To run `cudf` tests:
```bash
cd $CUDF_HOME/python
pytest -v ${CUDF_HOME}/python/cudf/cudf/tests
pytest -v ${CUDF_HOME}/python/dask_cudf/dask_cudf/ # There are tests in both tests/ and io/tests/
pytest -v ${CUDF_HOME}/python/custreamz/custreamz/tests
```

#### `cudf` (Java):

- First, build the `libcudf` C++ library following the steps above

- Then, refer to the [Java README](java/README.md)

Done! You are ready to develop for the cuDF project. Please review the project's
[code formatting guidelines](#code-formatting).

## Debugging cuDF

### Building in debug mode from source

Follow the instructions to [build from source](#build-cudf-from-source) and add `-g` to the
`./build.sh` command.

For example:

```bash
./build.sh libcudf -g
```

This builds `libcudf` in debug mode which enables some `assert` safety checks and includes symbols
in the library for debugging.

All other steps for installing `libcudf` into your environment are the same.

### Debugging with `cuda-gdb` and `cuda-memcheck`

When you have a debug build of `libcudf` installed, debugging with the `cuda-gdb` and
`cuda-memcheck` is easy.

If you are debugging a Python script, run the following:

```bash
cuda-gdb -ex r --args python <program_name>.py <program_arguments>
```

```bash
compute-sanitizer --tool memcheck python <program_name>.py <program_arguments>
```

### Device debug symbols

The device debug symbols are not automatically added with the cmake `Debug` build type because it
causes a runtime delay of several minutes when loading the libcudf.so library.

Therefore, it is recommended to add device debug symbols only to specific files by setting the `-G`
compile option locally in your `cpp/CMakeLists.txt` for that file. Here is an example of adding the
`-G` option to the compile command for `src/copying/copy.cu` source file:

```cmake
set_source_files_properties(src/copying/copy.cu PROPERTIES COMPILE_OPTIONS "-G")
```

This will add the device debug symbols for this object file in `libcudf.so`.  You can then use
`cuda-dbg` to debug into the kernels in that source file.

## Code Formatting

### Using pre-commit hooks

cuDF uses [pre-commit](https://pre-commit.com/) to execute all code linters and formatters. These
tools ensure a consistent code format throughout the project. Using pre-commit ensures that linter
versions and options are aligned for all developers. Additionally, there is a CI check in place to
enforce that committed code follows our standards.

To use `pre-commit`, install via `conda` or `pip`:

```bash
conda install -c conda-forge pre-commit
```

```bash
pip install pre-commit
```

Then run pre-commit hooks before committing code:

```bash
pre-commit run
```

By default, pre-commit runs on staged files (only changes and additions that will be committed).
To run pre-commit checks on all files, execute:

```bash
pre-commit run --all-files
```

Optionally, you may set up the pre-commit hooks to run automatically when you make a git commit. This can be done by running:

```bash
pre-commit install
```

Now code linters and formatters will be run each time you commit changes.

You can skip these checks with `git commit --no-verify` or with the short version `git commit -n`.

### Summary of pre-commit hooks

The following section describes some of the core pre-commit hooks used by the repository.
See `.pre-commit-config.yaml` for a full list.

C++/CUDA is formatted with [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html).

[`doxygen`](https://doxygen.nl/) is used as documentation generator and also as a documentation linter.
In order to run doxygen as a linter on C++/CUDA code, run

```bash
./ci/checks/doxygen.sh
```

Python code runs several linters including [Black](https://black.readthedocs.io/en/stable/),
[isort](https://pycqa.github.io/isort/), and [flake8](https://flake8.pycqa.org/en/latest/).

cuDF also uses [codespell](https://github.com/codespell-project/codespell) to find spelling
mistakes, and this check is run as a pre-commit hook. To apply the suggested spelling fixes,
you can run  `codespell -i 3 -w .` from the repository root directory.
This will bring up an interactive prompt to select which spelling fixes to apply.

## Developer Guidelines

The [C++ Developer Guide](cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md) includes details on contributing to libcudf C++ code.

The [Python Developer Guide](https://docs.rapids.ai/api/cudf/stable/developer_guide/index.html) includes details on contributing to cuDF Python code.


## Attribution

Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
Portions adopted from https://github.com/dask/dask/blob/master/docs/source/develop.rst
