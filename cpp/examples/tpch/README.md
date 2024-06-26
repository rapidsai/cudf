# TPC-H Inspired Examples

Implementing the TPC-H queries using `libcudf`. 

## Data Generation

We leverage the data generator (wrapper around official TPC-H datagen) from [Apache Datafusion](https://github.com/apache/datafusion) for generating data in the form of Parquet files. 

### Requirements 

- Rust

### Steps

1. Clone the Datafusion repository.
```bash
git clone git@github.com:apache/datafusion.git
```

2. Run the data generator. The data will be placed in a `data/` subdirectory.
```bash
cd benchmarks/
./bench.sh data tpch

# for scale factor 10,
./bench.sh data tpch10
```

## Implementation Status

- [x] Q1

