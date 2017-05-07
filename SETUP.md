# PyGDF

GPU DataFrame implementation using Numba


# Setup minconda

Get Miniconda for your platform from https://conda.io/miniconda.html.

For example on 64-bit Linux, run:

```bash
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh
```

and follow the instructions to complete the installation.

# Conda environments

Create testing environment with:

```bash
$ conda env create --file conda_environments/testing_py35.yml
```

This environment spec contains the exact version of each dependency.

# Testing

This project uses `py.test`

In the source root directory, run:

```bash
$ py.test
```

# Notebooks

See README in `./notebooks`


