#!/bin/bash

echo Please have a CUDA-compatible GPU, NVIDIA Drivers and CUDA installed beforehand. 

echo Installing Anaconda3
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh
source ~/.bashrc

echo Updating Anaconda
conda update -n base -c defaults conda  

echo Creating cuStreamz environment and activating it
conda create --name cuStreamz
source activate cuStreamz

echo Installing ipywidgets, ipykernel, python-confluent-kafka, ujson, and Dask
conda install -y -c anaconda ipywidgets 
conda install -y -c anaconda ipykernel
conda install -y -c conda-forge python-confluent-kafka
conda install -y -c anaconda ujson
conda install -y -c anaconda dask

echo Installing streamz
git clone https://github.com/python-streamz/streamz.git 
cd ./streamz && python setup.py build && python setup.py install

echo \>\>\>Please enter the cudf version you need \(we recommend 0.8\)
read cudf_version
echo \>\>\>Please enter the Python verison you have installed \(either 3.6 or 3.7\)
read py_version
echo \>\>\>Please enter the CUDA version you have installed \(either 9.2 or 10.0\)
read cuda_version

echo Installing cudf v$cudf_version
conda install -y -c nvidia -c rapidsai -c numba -c conda-forge -c defaults \
cudf=$cudf_version python=$py_version cudatoolkit=$cuda_version

echo All done! Please activate the cuStreamz environment ...
source activate cuStreamz
