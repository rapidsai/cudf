# strings_udf
User Defined Function (UDF) prototype for operating against a libcudf strings column.

UDFs are custom kernel/device CUDA code that execute in parallel threads on an NVIDIA GPU.
The libcudf library contains fixed operation primitives written in CUDA/C++ for strings.
Strings are variable length and libcudf is optimized to minimize costly memory allocations
when operations modify strings. This pattern can make custom UDF logic very difficult to write.

## Strings UDF device library

To make it easier to write UDFs, a device string class [dstring](cpp/include/cudf/strings/udf/dstring.hpp) has been created to perform specific operations on
individual strings. 
The libcudf strings column strings are read-only but an individual `cudf::string_view` instance can be copied to a `dstring` and this copy can be modified using the `dstring` methods.
Once the UDF has been executed on each string, the resulting strings
can be converted back into a libcudf strings column. 
Note that `dstring` uses CUDA malloc/free in device code to manage its string data.
We are hoping to improve this high cost memory allocation and deallocation in the future.

## Dependencies
The libcudf strings implementation is available here: https://github.com/rapidsai/cudf.
To build the artifacts in this repo requires an existing developer install of libcudf.
This means libcudf and its appropriate dependencies must be available for include and link from the CUDA compiler (nvcc).
Further, it is expected cuDF and libcudf are installed in a Conda
environment and there is a `CONDA_PREFIX` environment variable set to that conda environment directory.

Follow the [instructions to install cudf](https://github.com/rapidsai/cudf/#conda)

The CUDA toolkit must be installed and include [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html) to build and launch the UDFs.
The code here has only been tested on CUDA 11.5 but may work on 11.x or later versions as well.

## Building
Make sure the `CONDA_PREFIX` environment variable points to the conda environment directory
containing cuDF and libcudf (e.g. `/conda/env/rapids`).

### CUDA/C++
The CUDA/C++ code is built using `cmake` which is expected to be version `3.20.1` or higher.

From the `cpp` directory create a `build` directory and run `cmake` as follows:
```
cd cpp
mkdir build
cd build
cmake .. -DCONDA_PREFIX=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make
make install
```

This will build the `libcudf_strings_udf.so` functions library and the `udf_cli` command-line interface.

### Python
The `libcudf_strings_udf.so` must be built first since the Python library here depends on it for executing UDFs.
Follow the instructions in the [Python](python) directory to build the python library.


## Command Line Interface (CLI)

A CLI is included here to demonstrate executing a UDF on a libcudf strings column instance.
The strings column is by created from a text file with new-line delimiters for each string or from a CSV file. 
The cudf cuIO CSV reader is used to load the text file into a libcudf strings column.

The UDF CUDA kernel is launched with the number of threads equal to the number of input strings (rows). 
And each thread can work on a single row to produce a single output string.
After the threads complete, the resulting strings are gathered into a libcudf strings column.

Create the UDF function in a text file to pass to the CLI.
The output result can be written to a specified file or to the console.

The CLI parameters are as follows

| Parameter | Description | Default |
|:---------:| ----------- | ------- |
| -u        | Text file contain UDF kernel function in CUDA/C++ | (required) |
| -n        | Kernel function name in the UDF file | "udf_kernel" |
| -t        | Text or CSV-file for creating strings column | (required) |
| -i        | Include directory for libcudf | default is $CONDA_PREFIX/include |
| -c        | 0-based column number if CSV file | 0 (first column) |
| -r        | Number of rows to read from file | 0 (entire file) |
| -f        | Output file name/path | default output is stdout |
| -m        | Maximum malloc heap size in MB | 1000 |

Note that to run this CLI you might need to add paths to your `LD_LIBRARY_PATH`. For example:
```
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```
Note this also prepends the `.` path so that the current directory is included when locating dependent shared objects to load.

### Example
The following example shows parsing an XML file to retrieve the `name` attribute of each `entry`.

Here is the example XML file [`example.xml`](cpp/sample_udfs/example.xml)
```
<xml version='1.0'><base>
<entry age="8" name="Toby"/>
<entry address="283" name="David"/>
<entry name="Bill" size="10" />
<entry age="35" address="211" name="Ted" size="10"/>
<entry name="Jane" zip="90210" /></base>
```

Example UDF file [`xml.udf`](cpp/sample_udfs/xml.udf)
```
#include <cudf/strings/udf/dstring.cuh>

__device__ cudf::string_view find_attr( cudf::string_view const& input, 
                                        cudf::string_view const& name )
{
  auto pos = input.find(name);
  if( pos < 0 ) return cudf::string_view{"",0};

  cudf::string_view quote("\"",1);
  auto begin = input.find(quote, pos) + 1;
  auto end   = input.find(quote, begin);

  return input.substr(begin, end-begin);
}

__global__ void udf_kernel( cudf::string_view* d_in_strs, int size,
                            cudf::strings::udf::dstring* d_out_strs, int count )
{
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
  if( tid >= count ) return;

  auto input_str = d_in_strs[tid];
  
  auto name = find_attr(input_str, cudf::string_view("name",4));

  d_out_strs[tid] = name; // makes a copy here
}
```

Example CLI:
```
$ ./udf_cli -u ../sample_udfs/xml.udf -t ../sample_udfs/example.xml

Toby
David
Bill
Ted
Jane
```
