#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
# Backup environment variables (only if the variables are set)
if [[ ! -z "${CUDA_HOME+x}" ]]
then
  export CUDA_HOME_CONDA_NVCC_BACKUP="${CUDA_HOME:-}"
fi

if [[ ! -z "${CUDA_PATH+x}" ]]
then
  export CUDA_PATH_CONDA_NVCC_BACKUP="${CUDA_PATH:-}"
fi

if [[ ! -z "${CFLAGS+x}" ]]
then
  export CFLAGS_CONDA_NVCC_BACKUP="${CFLAGS:-}"
fi

if [[ ! -z "${CPPFLAGS+x}" ]]
then
  export CPPFLAGS_CONDA_NVCC_BACKUP="${CPPFLAGS:-}"
fi

if [[ ! -z "${CXXFLAGS+x}" ]]
then
  export CXXFLAGS_CONDA_NVCC_BACKUP="${CXXFLAGS:-}"
fi

if [[ ! -z "${CMAKE_ARGS+x}" ]]
then
  export CMAKE_ARGS_CONDA_NVCC_BACKUP="${CMAKE_ARGS:-}"
fi

# Default to using $(cuda-gdb) to specify $(CUDA_HOME).
if [[ -z "${CUDA_HOME+x}" ]]
then
    CUDA_GDB_EXECUTABLE=$(which cuda-gdb || exit 0)
    if [[ -n "$CUDA_GDB_EXECUTABLE" ]]
    then
        CUDA_HOME=$(dirname $(dirname $CUDA_GDB_EXECUTABLE))
    else
        echo "Cannot determine CUDA_HOME: cuda-gdb not in PATH"
        return 1
    fi
fi

if [[ ! -d "${CUDA_HOME}" ]]
then
    echo "Directory specified in CUDA_HOME(=${CUDA_HOME}) doesn't exist"
    return 1
fi

if [[ ! -f "${CUDA_HOME}/lib64/stubs/libcuda.so" ]]
then
    echo "File ${CUDA_HOME}/lib64/stubs/libcuda.so doesn't exist"
    return 1
fi

if [[ -z "$(${CUDA_HOME}/bin/nvcc --version | grep "Cuda compilation tools, release 11.5")" ]]
then
  echo "Version of installed CUDA didn't match package"
  return 1
fi

export CUDA_HOME="${CUDA_HOME}"
export CFLAGS="${CFLAGS} -isystem ${CUDA_HOME}/include"
export CPPFLAGS="${CPPFLAGS} -isystem ${CUDA_HOME}/include"
export CXXFLAGS="${CXXFLAGS} -isystem ${CUDA_HOME}/include"

### CMake configurations

# CMake looks up components in CUDA_PATH, not CUDA_HOME
export CUDA_PATH="${CUDA_HOME}"
# New-style CUDA integrations in CMake
CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDAToolkit_ROOT=${CUDA_HOME}"
# Old-style CUDA integrations in CMake
## See https://github.com/conda-forge/nvcc-feedstock/pull/58#issuecomment-752179349
CMAKE_ARGS+=" -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}"
## Avoid https://github.com/conda-forge/openmm-feedstock/pull/44#issuecomment-753560234
## We need CUDA_HOME in _front_ of CMAKE_FIND_ROOT_PATH
CMAKE_ARGS="$(echo ${CMAKE_ARGS} | sed -E -e "s|(-DCMAKE_FIND_ROOT_PATH=)(\S+)|\1$CUDA_HOME;\2|")"
export CMAKE_ARGS="${CMAKE_ARGS}"

### /CMake configurations

mkdir -p "${CONDA_BUILD_SYSROOT}/lib"
mkdir -p "${CONDA_PREFIX}/lib/stubs"

# Add $(libcuda.so) shared object stub to the compiler sysroot.
# Needed for things that want to link to $(libcuda.so).
# Stub is used to avoid getting driver code linked into binaries.


if [[ "${CONDA_BUILD}" == 1 ]]
then
  # Make a backup of $(libcuda.so) if it exists
  if [[ -f "${CONDA_BUILD_SYSROOT}/lib/libcuda.so" ]]
  then
    LIBCUDA_SO_CONDA_NVCC_BACKUP="${CONDA_BUILD_SYSROOT}/lib/libcuda.so-conda-nvcc-backup"
    mv "${CONDA_BUILD_SYSROOT}/lib/libcuda.so" "${LIBCUDA_SO_CONDA_NVCC_BACKUP}"
  fi
  ln -s "${CUDA_HOME}/lib64/stubs/libcuda.so" "${CONDA_BUILD_SYSROOT}/lib/libcuda.so"
else
  ln -sf "${CUDA_HOME}/lib64/stubs/libcuda.so" "${CONDA_PREFIX}/lib/stubs/libcuda.so"
fi

