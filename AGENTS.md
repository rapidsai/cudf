# AGENTS.md - cuDF Development Guide

cuDF is a GPU-accelerated DataFrame library providing C++ (libcudf), Python (cudf, pylibcudf),
and related packages for columnar data processing. The C++ library (libcudf) provides
GPU-accelerated data-parallel algorithms for column-oriented tabular data. The Python library
(cudf) provides a pandas-like API backed by libcudf.

## Safety Rules for Agents

- **Minimal diffs**: Change only what's necessary; avoid drive-by refactors.
- **No mass reformatting**: Don't run formatters over unrelated code.
- **No API invention**: Align with existing cuDF patterns and documented APIs.
- **Don't bypass CI**: Don't suggest skipping checks or using `--no-verify`.
- **CUDA/GPU hygiene**: Keep operations stream-ordered, use RMM allocators (never use raw
  `cudaMalloc` or other CUDA APIs for device memory). All libcudf APIs should accept
  a stream `rmm::cuda_stream_view` if they launch stream-ordered work and
  a memory resource `rmm::device_async_resource_ref` if they return memory to the caller.

### Before Finalizing a Change

Ask yourself:
- What scenarios must be covered? (happy path, edge cases, failure modes)
- What's the expected behavior contract? (inputs/outputs, errors)
- Where should tests live? (C++ gtests under `cpp/tests/`, Python pytest under the
  appropriate `python/<package>/` test directory)

## Build Commands

### Devcontainer (username: coder)

Note: `-j0` means "use all available CPU cores." This should be used for all builds in devcontainers.

#### Build dependency chain

Packages must be built in this order (each depends on the previous):
1. `build-cudf-cpp` — C++ library (libcudf)
2. `build-libcudf-python` — Python wheel wrapper for the C++ library
3. `build-pylibcudf-python` — Cython bindings (compiles `.pyx` files)
4. `build-cudf-python` / `build-cudf-polars-python` — high-level Python packages

`build-cudf -j0` runs the full chain automatically. When rebuilding a single package,
ensure its dependencies are already built.

#### Build commands
```bash
build-cudf-cpp -j0              # Build libcudf C++ library, accepts CMake flags (-D...)
build-libcudf-python -j0        # Build libcudf Python wrapper (needed before pylibcudf)
build-pylibcudf-python -j0      # Build pylibcudf Cython bindings
build-cudf-python -j0           # Build cudf Python package
build-cudf-polars-python -j0    # Build cudf-polars Python package (pure Python, fast)
build-cudf -j0                  # Build everything: C++ + all Python packages
```

#### Configure and clean commands
```bash
configure-cudf-cpp              # Re-run CMake configure without rebuilding
clean-cudf-cpp                  # Clean C++ build artifacts
clean-pylibcudf-python          # Clean pylibcudf build artifacts and uninstall
clean-cudf-polars-python        # Clean cudf-polars build artifacts and uninstall
clean-cudf                      # Clean all cudf build artifacts
```

### Standard Environment (requires conda env from conda/environments/)
```bash
./build.sh libcudf                  # Build and install C++ library
./build.sh libcudf -g               # Debug build
./build.sh libcudf tests            # Build with tests
./build.sh libcudf benchmarks       # Build with benchmarks
./build.sh clean libcudf            # Clean rebuild
./build.sh pylibcudf                # Build pylibcudf Python package
./build.sh cudf                     # Build cudf Python package
./build.sh cudf_polars              # Build cudf_polars Python package
./build.sh dask_cudf                # Build dask_cudf Python package
./build.sh libcudf pylibcudf cudf   # Build C++ and main Python packages
./build.sh --pydevelop libcudf pylibcudf cudf  # Build with Python in editable mode
```

### CMake Direct
```bash
cmake -S cpp -B cpp/build -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j$(nproc)
```

## Test Commands

### Devcontainer (username: coder)
```bash
test-cudf-cpp -j20             # Run all C++ tests (20 workers typical), accepts ctest flags
test-cudf-python -n8           # Run cudf Python tests (8 workers typical), accepts pytest flags
test-pylibcudf-python          # Run pylibcudf tests (runs 2 passes: with/without stream testing)
test-cudf-polars-python        # Run cudf-polars tests (runs 4 passes — see note below)
```

`test-cudf-polars-python` runs 4 separate pytest passes:
1. `--executor in-memory` — in-memory executor
2. `--executor streaming` — streaming executor
3. `--executor streaming --blocksize-mode small` — streaming with small blocksize
4. `tests/experimental --executor streaming --cluster distributed` — distributed cluster

For faster targeted testing, use pytest directly (see below) with the flags you need.

### C++ Tests (GoogleTest)
```bash
# Run all C++ tests
ctest --test-dir cpp/build --output-on-failure

# Run single test by name pattern
ctest --test-dir cpp/build -R copying

# Run specific test executable directly
./cpp/build/gtests/COPYING_TEST

# Run specific test case within executable
./cpp/build/gtests/COPYING_TEST --gtest_filter="CopyTest.*"
```

### Python Tests (pytest)
```bash
# cudf tests
pytest python/cudf/cudf/tests/ -v
pytest python/cudf/cudf/tests/test_dataframe.py -v
pytest python/cudf/cudf/tests/test_dataframe.py::test_specific_function -v

# pylibcudf tests
pytest python/pylibcudf/tests/ -v

# cudf_polars tests (must specify --executor; run from python/cudf_polars/)
cd python/cudf_polars
pytest tests/ -v --executor in-memory                            # In-memory executor
pytest tests/ -v --executor streaming                            # Streaming executor
pytest tests/ -v --executor streaming --blocksize-mode small     # Streaming, small blocks
pytest tests/test_scan.py -v --executor in-memory                # Single test file
pytest tests/test_scan.py::test_specific -v --executor in-memory # Single test

# dask_cudf tests
pytest python/dask_cudf/dask_cudf/tests/ -v
```

## Lint and Format

Always use pre-commit to run linters and formatters:
```bash
pre-commit run --all-files     # Run all hooks (recommended)
```

## Code Style Guidelines

Use `pre-commit run --all-files` to run linter and style checks. It will call clang-format,
ruff, and other tools.

### Naming Conventions
- **C++ classes/functions/methods**: `snake_case` (e.g., `column_view`, `gather`)
- **C++ macros**: `SCREAMING_SNAKE_CASE`
- **C++ template params**: `PascalCase` (e.g., `IteratorType`)
- **C++ private members**: prefixed with underscore (e.g., `_column`)
- **Python**: `snake_case` for functions and variables; `PascalCase` for classes

### Error Handling
- **C++**: Use cuDF exception types from `<cudf/utilities/error.hpp>`
  - `cudf::logic_error` for programming errors / precondition violations
  - `cudf::cuda_error` for CUDA runtime errors
  - `cudf::data_type_error` for unsupported dtype operations
  - Use `CUDF_EXPECTS()` macro for precondition checks
  - Use `CUDF_FAIL()` for unreachable/erroneous code paths
  - Use `CUDF_CUDA_TRY()` for CUDA API calls

### File Headers (SPDX format, required)

C++ and CUDA:
```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
```

Python:
```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
```

### Documentation
- **C++**: Doxygen comments for public APIs (`/** ... */`). Run `./ci/checks/doxygen.sh`
  to lint.
- **Python**: NumPy-style docstrings
- All public functions must be documented

## Project Structure
```
cpp/                              # C++ source code (libcudf)
├── include/cudf/                 # Public C++ headers
├── include/cudf/detail/          # Internal detail headers
├── src/                          # Implementation files
├── tests/                        # GoogleTest tests
├── benchmarks/                   # Google Benchmark benchmarks
└── doxygen/developer_guide/      # C++ developer guide docs
python/
├── pylibcudf/                    # Low-level Python/Cython bindings
│   └── tests/                    # pylibcudf tests
├── cudf/                         # Main cudf Python package (pandas-like API)
│   └── cudf/tests/               # cudf tests
├── cudf_polars/                  # Polars GPU engine via cudf
│   └── tests/                    # cudf_polars tests
├── dask_cudf/                    # Dask integration
│   └── dask_cudf/tests/          # dask_cudf tests
├── cudf_kafka/                   # Kafka integration
├── custreamz/                    # Streaming integration
└── libcudf/                      # Python build wrapper for libcudf
java/                             # Java bindings
ci/                               # CI scripts
```

## Developer Guide References

For detailed C++ development guidance, see the docs in `cpp/doxygen/developer_guide/`:
- **[DEVELOPER_GUIDE.md](cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md)**: Comprehensive
  C++ contributor guide covering directory structure, naming conventions, libcudf data
  structures (columns, tables, scalars, views), memory and stream conventions, public/detail
  API patterns, and more.
- **[TESTING.md](cpp/doxygen/developer_guide/TESTING.md)**: C++ unit test guidelines
  including test fixtures, typed tests, column wrappers, column comparison utilities, and
  stream validation.
- **[DOCUMENTATION.md](cpp/doxygen/developer_guide/DOCUMENTATION.md)**: Doxygen
  documentation guidelines.
- **[BENCHMARKING.md](cpp/doxygen/developer_guide/BENCHMARKING.md)**: Benchmarking
  guidelines.
- **[PROFILING.md](cpp/doxygen/developer_guide/PROFILING.md)**: Profiling guidelines.

For the Python developer guide, see
[docs.rapids.ai](https://docs.rapids.ai/api/cudf/stable/developer_guide/index.html).

For general contribution workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

## PR Requirements
- All tests must pass
- Pre-commit checks must pass
- Update documentation for API changes
- Add tests for new functionality
- C++ changes require at least 2 approvals from cudf-cpp-codeowners

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Main build script (never used in devcontainers) | `build.sh` |
| CMake configuration | `cpp/CMakeLists.txt` |
| C++ public headers | `cpp/include/cudf/` |
| C++ detail headers | `cpp/include/cudf/detail/` |
| Error handling macros | `cpp/include/cudf/utilities/error.hpp` |
| Type utilities | `cpp/include/cudf/utilities/traits.hpp`, `type_dispatcher.hpp` |
| C++ test utilities | `cpp/include/cudf_test/` (column wrappers, comparison macros, etc.) |
| C++ tests | `cpp/tests/` |
| C++ benchmarks | `cpp/benchmarks/` |
| pylibcudf (Cython bindings) | `python/pylibcudf/` |
| cudf Python package | `python/cudf/` |
| cudf_polars Python package | `python/cudf_polars/` |
| CI configuration | `ci/` |

## Resources

- **Documentation**: https://docs.rapids.ai/api/cudf/stable/
- **GitHub Issues**: https://github.com/rapidsai/cudf/issues
