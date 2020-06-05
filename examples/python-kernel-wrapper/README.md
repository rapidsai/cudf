# Python Kernel Wrapper

## Overview
This example demonstrates how to share cudf dataframes between Python and custom CUDA kernels. This is useful for performing custom CUDA-accelerated business logic on cuDF dataframes and handling certain tasks in Python and others in CUDA.

Dataframes that are created in Python cuDF are already present in GPU memory and accessible to CUDA code. This makes it straightforward to write a CUDA kernel to work with a dataframe columns. In fact this is how libcudf processes dataframes in CUDA kernels; the only difference in this example is that we invoke CUDA kernels that exist outside the cuDF code base. The term User Defined Function (UDF) could be loosely used to describe what this example is demonstrating.

This example provides a Cython `kernel_wrapper` implementation to make sharing the dataframes between Python and our custom CUDA kernel easier. This wrapper allows Python users to seamlessly invoke those CUDA kernels with a single function call and also provides a clear place to implement the C++ "glue code".

The example CUDA kernel accepts a data column (PRCP) containing rainfall values stored as 1/10th of a mm and converts those values to inches. The dataframe is read from a local CSV file using Python. Python then invokes the CUDA mm->inches conversion kernel via the Cython `kernel_wrapper`, passing it the dataframe object. The converted data can then be accessed from Python, e.g. using `df.head()`.

This is similar to an existing [weather notebook](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.14/intermediate_notebooks/examples/weather.ipynb), which provides a reference for understanding the implementation. 

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
