# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cuDF is a GPU DataFrame library that provides a pandas API accelerated by NVIDIA GPUs. The repository contains multiple components:

- **libcudf**: Core C++/CUDA library implementing GPU-accelerated dataframe operations
- **cudf**: Python pandas-compatible API wrapping libcudf
- **pylibcudf**: Low-level Python bindings for libcudf
- **dask_cudf**: Dask integration for distributed GPU computing
- **cudf_kafka**: Kafka integration for streaming data
- **custreamz**: Streaming data processing utilities
- **cudf_polars**: Polars backend implementation

## Development Commands

To build each component, use the various `build-*` commands automatically added to the path.
```bash
build-cudf-cpp
build-pylibcudf-python
build-cudf-python
build-dask_cudf-python
build-cudf_kafka-python
build-custreamz-python
build-cudf_polars-python
```

You can also use a single `build-cudf` command to build all components.

### Testing Commands
```bash
# C++ tests
ctest --test-dir cpp/build/latest

# Python tests
cd python
pytest python/cudf/cudf/tests                            # cudf tests
pytest python/dask_cudf/dask_cudf/                       # dask_cudf tests
```

### Code Quality Commands
All hooks should always be run using pre-commit to ensure consistent environments.
```bash
# Pre-commit hooks (handles formatting, linting)
pre-commit run                    # On staged files
pre-commit run --all-files        # On all files
```

## Architecture Overview

### Repository Structure
```
├── cpp/                    # C++/CUDA implementation
│   ├── include/cudf/      # Public C++ headers
│   ├── src/               # C++ source code organized by functionality
│   ├── tests/             # C++ unit tests
│   └── benchmarks/        # Performance benchmarks
├── python/                # Python packages
│   ├── cudf/              # Main Python DataFrame API
│   ├── pylibcudf/         # Low-level Python bindings
│   ├── dask_cudf/         # Dask integration
│   ├── cudf_kafka/        # Kafka integration
│   ├── custreamz/         # Streaming utilities
│   └── cudf_polars/       # Polars backend
├── java/                  # Java bindings
└── ci/                    # Continuous integration scripts
```

### Core C++ Library (libcudf)

The C++ library is organized into functional modules:
- **column**: Core columnar data structures and operations
- **table**: Multi-column data structures
- **io**: File format readers/writers (Parquet, ORC, CSV, JSON, Avro)
- **aggregation**: GroupBy and reduction operations
- **sorting**: Sorting and ordering operations
- **join**: Join operations (inner, outer, semi, anti)
- **strings**: String processing operations
- **datetime**: Date/time operations
- **copying**: Data copying and slicing operations
- **binaryop**: Element-wise binary operations
- **unary**: Element-wise unary operations
- **groupby**: GroupBy aggregation operations

### Python Package Structure

**cudf**: Main user-facing package providing pandas-compatible API
- Wraps libcudf functionality through pylibcudf bindings
- Implements DataFrame, Series, Index classes
- Provides GPU memory management through RMM

**pylibcudf**: Low-level Cython bindings
- Direct interface to C++ libcudf functions
- Minimal Python overhead for performance-critical operations
- Used as foundation by higher-level cudf package

## Key Dependencies

### C++ Dependencies
- CUDA 12.0+ with compute capability 7.0+
- CMake 3.29.6+
- GCC 13.3+
- Apache Arrow for columnar data format
- RMM for GPU memory management

### Python Dependencies
- PyArrow for interoperability
- Numba for JIT compilation
- CuPy for GPU array operations
- RMM for memory management
- Pandas for API compatibility

## Build System

- **CMake**: Primary build system for C++ components
- **Rapids-CMake**: RAPIDS-specific CMake utilities and configurations
- **scikit-build-core**: Python package build backend
- **conda**: Primary dependency and environment management

The build system supports:
- Multi-GPU architecture compilation
- Debug and release builds
- Static and shared library configurations
- NVTX profiling integration
- Per-thread default CUDA streams

## Development Guidelines

### Code Organization
- C++ code follows libcudf module organization
- Python code mirrors pandas API structure where possible
- Tests are co-located with source code
- Benchmarks are separate from functional code

### Memory Management
- All GPU allocations use RMM (RAPIDS Memory Manager)
- RAII patterns for automatic resource cleanup
- Column/table data is immutable by design
- Views provide zero-copy data access

### Error Handling
- C++ uses exceptions for error propagation
- CUDA errors are wrapped in cudf exceptions
- Python preserves pandas exception behavior where possible

### Performance Considerations
- Prefer columnar operations over row-wise processing
- Use views to avoid unnecessary data copying
- Leverage CUDA streams for concurrent operations
- Consider memory bandwidth in algorithm design
