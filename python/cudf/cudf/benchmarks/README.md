# cuDF benchmarks

## Overview

This directory contains source and configuration files for benchmarking
`cuDF`. The sources are currently intended to benchmark `cuDF` via the
python API, but this is not a requirement.

## Prerequisites
### Datasets
* Download datasets using `get_datasets.sh` shell file. Currently there are
  only avro and json datasets.

## Usage
### Python
* Run benchmarks using pytest as shown below

```
pytest cudf/benchmarks/
```
* cuIO benchmarks have option of using file path directly or memory buffers,
  by default file path option is enabled. To enable memory buffer usage, use
  `--use_buffer True` with pytest as shown below.
```
pytest --use_buffer True cudf/benchmarks/
```

* If the datasets directory is different from default, it can be
  provided with option "--dataset_dir directory_path" as shown below.
```
pytest --dataset_dir directory_path cudf/benchmarks/
```


