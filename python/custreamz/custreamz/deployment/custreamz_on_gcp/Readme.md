**cuStreamz GCP Setup**

*"The following instruction set is for setting up the cuStreamz pipeline on Google Cloud Platform (GCP). But, the overall process for deploying cuStreamz
on any CSP would be the same."*
— cuStreamz team

**Step 1: Create a cuStreamz VM instance** 
1. On the Compute Engine console of GCP, you can create a new VM instance. We suggest you use 24 cores, 1 T4 *OR* 48 cores, 2 T4s for the best results.
2. You can add SSH keys to this instance from the Security tab so that you can SSH into this VM instance any external SSH-client.
3. Once you launch this instance, check whether or not NVIDIA drivers and CUDA is properly installed. 
4. To verify that your GPU is CUDA-capable, use: `lspci | grep -i nvidia`. If this command does not show any devices, restart the instance or create it again from the beginning.
5. To check whether or not NVIDIA driver is installed and working properly, use: `nvidia-smi`. If this fails, do the following:
```
wget http://us.download.nvidia.com/tesla/410.79/NVIDIA-Linux-x86_64-410.79.run 
sudo bash NVIDIA-Linux-x86_64-410.79.run 
```
6. To verify CUDA installation, use: `nvcc --version`. We recommend you use CUDA 9.2 or higher. If nvidia-smi works and nvcc --version does not work, try running the following:
```
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
7. If you are using multiple GPUs, it's best to set CUDA_VISIBLE_DEVICES like: `export CUDA_VISIBLE_DEVICES=0,1` (in case of 2 GPUs).
8. Open ports 8888 and 8787 for the Jupyter endpoint for cuStreamz notebooks and Dask diagnostic dashboard respectively. On GCP, this can be done by adding the appropriate firewall rule in VPC Netowrk console.

**Step 2: Install cuStreamz requirements**
1. Install Anaconda (we recommend Anaconda3). Reference link: [https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)  
2. Create a conda environment for cuStreamz work, and add the environment as a kernel to work in Jupyter Notebooks. 
```
conda create —name cuStreamz
python -m ipykernel install --user --name cuStreamz --display-name "cuStreamz"
```
3. Activate the conda environment using: `conda activate cuStreamz`.
4. Install ipywidgets, ipykernel, python-confluent-kafka, dask, ujson. 
```
conda install ipywidgets
conda install ipykernel
conda install -c conda-forge python-confluent-kafka 
conda install dask
conda install ujson
```
5. Install streamz from GitHub using:
```
git clone https://github.com/python-streamz/streamz.git
cd streamz
python setup.py build
python setup.py install
```
6. Install cudf v0.8 from RAPIDS.ai, using specifications in: [https://rapids.ai/start.html](https://rapids.ai/start.html)
```
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults \
    cudf=0.8 python=3.7 cudatoolkit=10.0
```
7. Start a Jupyter endpoint on port 8888 for writing cuStreamz code using: `nohup jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root &`. 
8. If you face problems importing cudf, try setting the NUMBAPRO_NVVM and NUMBAPRO_LIBDEVICE environment variables using: 
```
export NUMBAPRO_NVVM=/usr/local/cuda-10.0/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-10.0/nvvm/libdevice/
```

**Step 3: Start the cuStreamz pipeline**
1. Use the dask_scheduler.py to start the Dask scheduler and workers. It would also open the Dask diagnostics dashboard on port 8787.
2. Use the cuStreamz_job.py to get started with a simple cuStreamz job.
3. One can produce random data to Kafka (assuming you have setup a Kafka cluster locally, with a single broker), using produce_random_to_kafka.py. 
4. For our experiments, we are using Kafka Cluster deployment provided by Bitnami from the Deployment Manager on GCP. One can try using that, too. 

**Step 4: You're now streaming!**  <br />
Please feel free to get in touch with us if you're facing any problems in the deployment process, and we'll try our best to look into them.
