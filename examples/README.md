# cuDF Examples

## Setup
A couple of assumptions are made about the setup for building and running this example.

1. Conda is installed on the build machine. Conda is used for ensuring dependencies are present in for building. While conda is not required it greatly reduces the complexity around wrangling the necessary dependencies.
2. CMake is used used for the build tool.
3. CUDA is installed. This version was tested with CUDA 10.2 but other versions should work.

### Steps
1. Clone the repo and submodules ```git clone https://github.com/jdye64/libcudf-examples.git && cd libcudf-examples && git submodule update --init --remote --recursive```
2. We need to create a conda environment. ```conda env create -f ./conda/cudf_dev_cuda10.2.yml --name cudf_dev```
3. Activate the conda environment ```conda activate cudf_ex```
3. Create the cmake env. ```mkdir ./cpp/build && cd ./cpp/build && cmake ..```
4. Build the C++ and CUDA executables and shared objects ```make```
5. Create the Python module using Cython ```cd python && python setup.py build_ext --inplace && python setup.py install --single-version-externally-managed --record=record.txt```
99. Download weather data. A convenience Python script has been provided here to make that easier for you. By default it will download years 2010-2020 weather data. That data is about 300MB per file so if you need to download less files you can change that in the script. The data will be downloaded to ${EXAMPLE_HOME}/data/weather. ```python ./data/download_data.py```
100. Run the executable. It expects an input of a single Weather year file and the output path for the resulting parquet file. EX: ```./cudf-examples /home/data/weather/2010.csv.gz /home/data/weather/results.parquet```