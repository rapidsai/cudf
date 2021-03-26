# Basic Standalone libcudf C++ application

This simple C++ example demonstrates a basic libcudf use case and provides a
minimal example of building your own application based on libcudf using CMake.

The example source code loads a csv file that contains stock prices from 4
companies spanning across 5 days, computes the average of the closing price
for each company and writes the result in csv format.

## How to compile and execute

Prerequisites:
- Cudatookit 11.0 or later
- gcc-9
- cmake 3.18
- `libboost-filesystem-dev` and `zlib1g-dev`

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/libcudf_example
```

Expect the first build to take some time, as it builds libcudf on the host
machine. It may be sped up by configuring the proper `PARALLEL_LEVEL` number.

We also provide a Dockerfile that helps setup the environment and automate
the build process.
