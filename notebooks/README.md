# cuDF Notebooks
## Intro
These notebooks provide examples of how to use cuDF.  These notebooks are designed to be self-contained with the `runtime` version of the [RAPIDS Docker Container](https://hub.docker.com/r/rapidsai/rapidsai/) and [RAPIDS Nightly Docker Containers](https://hub.docker.com/r/rapidsai/rapidsai-nightly) and can run on air-gapped systems.  You can quickly get this container using the install guide from the [RAPIDS.ai Getting Started page](https://rapids.ai/start.html#get-rapids)

## Notebooks
Notebook Title | Status | Description     
--- | --- | ---                                                                                                                                                                                           
[Numba-cuDF Integration](notebooks_numba_cuDF_integration.ipynb) | Working | A demonstration of GPU accelerated Python library interoperability, including a few examples of accelerating cuDF Dataframe operations using Numba kernels directly.

[Apple Operations in cuDF](notebooks_Apply_Operations_in_cuDF.ipynb) | Working | Accelerated, customized data transformation has been found to be very valuable. cuDF provides two special methods that serve this particular purpose: apply_rows and apply_chunks functions, which utilize the Numba library to accelerate the data transformation via GPU in parallel. This notebook shows a few examples of how to use them.

## RAPIDS notebooks
Visit the main RAPIDS [notebooks](https://github.com/rapidsai/notebooks) repo for a listing of all notebooks across all RAPIDS libraries.
