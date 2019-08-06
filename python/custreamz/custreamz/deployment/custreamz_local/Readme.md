**Deploying cuStreamz locally**

***This deployment process assumes you have a CUDA-compatible GPU, and have NVIDIA drivers and CUDA 10.0 installed. Please change the container tag in the Dockerfile according to the CUDA version you have installed.***

**Pre-requisite fulfillment** <br />
If you have a CUDA-compatible GPU, and have NVIDIA drivers and CUDA 10.0 installed, please skip this section.
1. To verify that your GPU is CUDA-capable, use: lspci | grep -i nvidia. If this command does not show any devices, restart the instance or create it again from the beginning.
2. To check whether or not NVIDIA driver is installed and working properly, use: nvidia-smi. If this fails, do the following:
```
wget http://us.download.nvidia.com/tesla/410.79/NVIDIA-Linux-x86_64-410.79.run 
sudo bash NVIDIA-Linux-x86_64-410.79.run 
```
3. To verify CUDA installation, use: nvcc --version. We recommend you use CUDA 9.2 or higher. If nvidia-smi works and nvcc --version does not work, try running the following:
```
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
4. If you are using multiple GPUs, it's best to set CUDA_VISIBLE_DEVICES like: `export CUDA_VISIBLE_DEVICES=0,1` (in case of 2 GPUs).

**Steps to start deployment — Dockerfile and Install script options.**
1. If you would like to use an install script instead of Docker, skip to step 3. 
2. Use Docker to build and then run the Dockerfile as an image. The image would install all the required libraries for deploying cuStreamz jobs end-to-end.
3. It would also open up a Jupyter Notebook endpoint on port 8888, the password for which would be 'cuStreamz'.
4. Instead of using the Dockerfile, another option is that can also just use `bash build.sh` and run the installation script. 
   This script installs everything required for cuStreamz development including Anaconda, and cudf based on the CUDA and Python versions you have installed. 
   Remember to select **no** when you are prompted 'Do you wish the installer to initialize Anaconda3 by running conda init?'
   It will also create a conda environment 'cuStreamz'. You can then just start a Jupyter notebook from the cuStreamz environment using `nohup jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 &`.
5. If you face problems with importing cudf, try setting the NUMBAPRO_NVVM and NUMBAPRO_LIBDEVICE environment variables using:
```
export NUMBAPRO_NVVM=/usr/local/cuda-10.0/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-10.0/nvvm/libdevice/
```
6. Use the dask_scheduler.py to start the Dask scheduler and workers. It would also open the Dask diagnostics dashboard on port 8787.
7. Use the cuStreamz_job.py to get started with a simple cuStreamz job.
8. One can produce random data to Kafka (assuming you have setup a Kafka cluster locally, with a single broker), using produce_random_to_kafka.py.
9. You're now streaming!

Please feel free to get in touch with us if you're facing any problems in the deployment process, and we'll try our best to look into them.