# PyGDF

[![Documentation Status](https://readthedocs.org/projects/pygdf/badge/?version=latest)](http://pygdf.readthedocs.io/en/latest/?badge=latest)

PyGDF implements the Python interface to access and manipulate the GPU Dataframe of [GPU Open Analytics Initialive (GOAI)](http://gpuopenanalytics.com/).  We aim to provide a simple interface that similar to the Pandas dataframe and hide the details of GPU programming.

## GPU open analytics initiative

Continuum Analytics, H2O.ai, and MapD Technologies have announced the formation of the GPU Open Analytics Initiative (GOAI) to create common data frameworks enabling developers and statistical researchers to accelerate data science on GPUs. GOAI will foster the development of a data science ecosystem on GPUs by allowing resident applications to interchange data seamlessly and efficiently.

![GOAI](img/goai_logo_3.png)

## GPU Data Frame

Our first project: an open source GPU Data Frame with a corresponding Python API. The GPU Data Frame is a common API that enables efficient interchange of data between processes running on the GPU. End-to-end computation on the GPU avoids transfers back to the CPU or copying of in-memory data reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. Users of the MapD Core database can output the results of a SQL query into the GPU Data Frame, which then can be manipulated by the Continuum Analytics’ Anaconda NumPy-like Python API or used as input into the H2O suite of machine learning algorithms without additional data manipulation.

![Architecture](img/GPU_df_arch_diagram.png)

Users of the MapD Core database can output the results of a SQL query into the GPU Data Frame, which then can be manipulated by the Continuum Analytics’ Anaconda NumPy-like Python API or used as input into the H2O suite of machine learning algorithms without additional data manipulation. In early internal tests, this approach exhibited order-of-magnitude improvements in processing times compared to passing the data between applications on a CPU.

![Architecture](img/mapd-conda-h2o.png)


## Setup

The following are instructions to setup a development environment.  We don't have release packages yet.

### Setup with Conda

You can get a minimal conda installation with [Miniconda](https://conda.io/miniconda.html).

For example on 64-bit Linux, run:

```bash
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh
```

and follow the instructions to complete the installation.

Once you have conda installed, create an environment with:

```bash
$ conda env create --name pygdf_dev --file conda_environments/testing_py35.yml
```

This creates an environment named "pygdf_dev" with the exact version of each dependency.

Activate the environment with:

```bash
$ source activate pygdf_dev
```
### Setup with Pip

Currently, we don't support pip install yet.  Please use conda for the time being.

### Testing

This project uses [py.test](https://docs.pytest.org/en/latest/).

In the source root directory and with the development environment activated, run:

```bash
$ py.test
```

### Demo notebooks

Please see [README](https://github.com/gpuopenanalytics/pygdf/blob/master/notebooks/README.md).

## Initial Committers

- Siu Kwan Lam
- Arno Candel
- Minggang Yu
- Stanley Seibert
- Jon Mckinney
- Bill Maimone
- Vinod Iyengar
- Todd Mostak
