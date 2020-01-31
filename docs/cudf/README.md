# Building Documentation

In order to build the docs, we need the conda dev environment from cudf and build cudf from source. 


# Steps to follow:

1. Create a conda env and  build cudf from source. The dependencies are installed to build rapids from source in that conda environment, then rapids is built and installed into the same environment.

2. Once cudf is built from source, navigate to "/cudf/docs/cudf/", i.e., `cd cudf/docs/cudf` and run makefile:

```bash
make html
```
Outputs to `build/html/index.html`


## View docs html page:

First navigate to "/build/html/" folder, i.e., `cd build/html` and then run the following command:

```
python -m http.server

```
Open a browser and type <host-ip>:8000. 

