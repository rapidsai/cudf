**Deploying cuStreamz locally**

***This deployment process assumes you have a CUDA-compatible GPU, and have NVIDIA drivers and CUDA 10.0 installed. Please change the container tag in the Dockerfile according to the CUDA version you have installed.***

**Pre-requisite fulfillment** <br />
If you have a CUDA-compatible GPU, and have NVIDIA drivers and CUDA 10.0 installed, please skip this section.
1. To verify that your GPU is CUDA-capable, use: 'lspci | grep -i nvidia'. If this command does not show any devices, restart the instance or create it again from the beginning.
2. To check whether or not NVIDIA driver is installed and working properly, use: 'nvidia-smi'. If this fails, please use the following link to install the correct CUDA-compatible drivers: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
3. To verify CUDA installation, use: 'nvcc --version'. We recommend you use CUDA 9.2 or 10.0. If nvidia-smi works and nvcc --version does not work, try running the following:
```
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

**Deployment using Dockerfile**
1. If you would like to use an install script instead of Docker, please skip to the next section.
2. Use Docker to build and then run the Dockerfile as an image. The image would install all the required libraries for deploying cuStreamz jobs end-to-end.
3. It would also open up a Jupyter Notebook endpoint on port 8888, the password for which would be 'cuStreamz'.

**Deployment using Install script**
1. Instead of using the Dockerfile, another option is that can also just use `bash build.sh` and run the installation script. 
   This script installs everything required for cuStreamz development including Anaconda, and cudf based on the CUDA and Python versions you have installed. 
   Remember to select **no** when you are prompted 'Do you wish the installer to initialize Anaconda3 by running conda init?'
   It will also create a conda environment 'cuStreamz'. You can then just start a Jupyter notebook from the cuStreamz environment using `nohup jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 &`.

**Running a cuStreamz pipeline end-to-end**
1. Use the dask_scheduler.py to start the Dask scheduler and workers. It would also open the Dask diagnostics dashboard on port 8787.
2. Use the cuStreamz_job.py to get started with a simple cuStreamz job.
3. One can produce random data to Kafka (assuming you have setup a Kafka cluster locally, with a single broker), using produce_random_to_kafka.py.
4. You're now streaming!

Please feel free to get in touch with us if you're facing any problems in the deployment process, and we'll try our best to look into them.
