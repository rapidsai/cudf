# Building Documentation

In order to build the docs, we need the conda dev environment from cudf and build cudf from source. 


## Steps to follow:

1. Create a conda env and  build cudf from source. The dependencies are installed to build rapids from source in that conda environment, then rapids is built and installed into the same environment.

2. Once cudf is built from source, navigate to "/cudf/docs/cudf/", i.e., `cd cudf/docs/cudf`. If you have your documentation written and want to turn it into HTML, run makefile:

```bash
make html
```
This should run Sphinx in your shell, and outputs to `build/html/index.html`.


## View docs web page by opening HTML in browser:

First navigate to "/build/html/" folder, i.e., `cd build/html` and then run the following command:

```
python -m http.server
```
Then, navigate a web browser to the IP address or hostname of the host machine at port 8000:

```
https://<host IP-Address>:8000
```
Now you can check if your docs edits formatted correctly. 
