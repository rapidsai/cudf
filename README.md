# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuDF - A GPU-accelerated DataFrame library for tablular data processing</div>

cuDF (pronounced "KOO-dee-eff") is an Apache 2.0 Licenced, GPU-accelerated DataFrame library
for tabular data processing. cuDF is one library apart of the [RAPIDS](https://rapids.ai/) GPU
Accelerated Data Science suite of libraries.

cuDF is composed of multiple libraries including:

* [libcudf](https://docs.rapids.ai/api/cudf/stable/libcudf_docs/): A CUDA C++ library with [Apache Arrow](https://arrow.apache.org/) compliant
data structures and fundamental algorithms for tabular data.
* [pylibcudf](https://docs.rapids.ai/api/cudf/stable/pylibcudf/): A Python library providing [Cython](https://cython.org/) bindings for libcudf.
* [cudf](https://docs.rapids.ai/api/cudf/stable/user_guide/): A Python library providing
    - A DataFrame library mirroring the [pandas](https://pandas.pydata.org/) API
    - A zero-code change accelerator, [cudf.pandas](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/), for existing pandas code.
* [cudf-polars](https://docs.rapids.ai/api/cudf/stable/cudf_polars/): A Python library proving a GPU engine for [Polars](https://pola.rs/)
* [dask-cudf](https://docs.rapids.ai/api/dask-cudf/stable/): A python library providing a GPU backend for [Dask](https://www.dask.org/) DataFrames

Notable external projects that use cuDF include:

* [spark-rapids](https://github.com/NVIDIA/spark-rapids): A GPU accelerator plugin for [Apache Spark](https://spark.apache.org/)
* [Velox-cuDF](https://github.com/facebookincubator/velox/blob/main/velox/experimental/cudf/README.md): A [Velox](https://velox-lib.io/)
extension module to execute Velox plans on the GPU
* [Sirius](https://www.sirius-db.com/): A GPU-native SQL engine providing extensions for libraries like [DuckDB](https://duckdb.org/)

## Installation

Please visit the [RAPIDS Installation Guide](https://docs.rapids.ai/install/) for details regarding how to install a stable or nightly release
of cuDF on your local workstation or the cloud using Docker, pip, conda, and more.

To install cuDF from source, please follow [the contribution guide](CONTRIBUTING.md#setting-up-your-build-environment) detailing
how to setup the build environment.

## Questions and Discussion

For bug reports or feature requests, please [file an issue](https://github.com/pandas-dev/pandas/issues/new/choose) on the Github issue tracker.

For questions or discussion about cuDF and GPU data processing, feel free to post in the [RAPIDS Go-Ai Slack](https://rapids-goai.slack.com/join/shared_invite/zt-trnsul8g-Sblci8dk6dIoEeGpoFcFOQ#/shared-invite/email) workspace.

## Contributing

cuDF is open to contributions from the community! Please see our [guide for contributing to cuDF](CONTRIBUTING.md) for more information.
