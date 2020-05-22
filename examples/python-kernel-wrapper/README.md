# Python Kernel Wrapper

## Overview
This examples aims to demonstrate how its possible to share cudf dataframes between Python and CUDA Kernels. This is useful if either an existing CUDA codebase is present to solve business logic or if teams would like the ability to handle certain tasks in Python and others in CUDA.

Since cudf uses libcudf, dataframes that are created in Python are already present on the GPU memory and available to the C++ codebase. This makes it fairly straightforward to write a CUDA kernel for working with a dataframe column. In fact this is how libcudf processes dataframes in CUDA kernels within cudf itself, the only difference in this example is we are writing CUDA kernels that are not in the cudf codebase. The term User Defined Function (UDF) could be loosely used to describe what this example is demonstrating.

A cython ```kernel_wrapper``` implementation has been created in this example to further make sharing the dataframes between Python and your CUDA kernel "UDF" easier. This wrapper allows for Python users to seamless invoke those "UDF" CUDA kernels with a single function call and also provides a clear place for implementing the C++ "glue code".

Now that we understand the motivation and how it will work lets define the problem for this example.

The goal is to write a CUDA kernel that accepts a rainfall (PRCP) data column and convert those values to inches. The original data is stored as 1/10th of a mm so the kernel must convert those values to inches. The dataframe will be read from a local csv file and created using Python. Python will then invoke the CUDA kernel doing the mm->inches conversion with the ```kernel_wrapper``` and the dataframe object. Once the kernel has finished its execution the dataframe values will be changed on the Python side as well so those can be seen by simple output with ```df.head()```.

This is similar to what is being done in an existing [weather notebook](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.14/intermediate_notebooks/examples/weather.ipynb) so if any confusion arises that is a good place to reference for general understanding. 

## Building

### Assumptions
1. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install) is installed.
2. [CUDA](https://developer.nvidia.com/cuda-downloads) 10.0 > is installed and on the PATH.

## Build Steps
1. Clone the cudf repo if you haven't already ```git clone https://github.com/rapidsai/cudf.git && cd cudf/examples/python-kernel-wrapper```
2. Create the ```cudf_ex``` conda environment. ```conda env create -f ./conda/cudf_ex.yml --name cudf_ex```
3. Activate the ```cudf_ex``` sconda environment ```conda activate cudf_ex```
3. Build the cython "kernel_wrapper" code. ```cd cython && python setup.py build install``` Notice the custom Kernel definitions are in ```cython/src/kernel.cu```. This is just for example and the build can be altered as needed.
4. Download weather data. A convenience Python script has been provided here to make that easier for you. By default it will download years 2010-2020 weather data. That data is about 300MB per file so if you need to download less files you can change that in the script. The data will be downloaded to ./data/weather. ```python ./data/download_data.py```
5. Run the Python example script. It expects an input of a single Weather year file. EX: ```python ./01_python_kernel_wrapper.py ./data/weather/2010.csv.gz```
