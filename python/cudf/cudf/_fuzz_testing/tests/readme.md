# Fuzz Tests

This directory contains all the Fuzz tests for cudf library.


## Steps to write a fuzz test

1. Add a Data Handler class which actually generates the necessary random data according to your requirements. This class should be added in `cudf/cudf/testing/`. A sample data handler class is: `CSVWriter`: https://github.com/rapidsai/cudf/blob/branch-0.16/python/cudf/cudf/testing/csv.py
2. Data Handlers are registered by the `pythonfuzz` decorator. At runtime, the Fuzzer will continuously run registered fuzz tests.
  
```python
from cudf.testing.csv import CSVWriter

@pythonfuzz(data_handle=CSVWriter)
def csv_writer_test(data_from_generate_input):
    ...
    ...
    ...

if __name__ == "__main__":
    ...
    ...

```
## Steps to run fuzz tests

1. To run a fuzz test, for example a test(method) is in `write_csv.py`:

```bash
python write_csv.py your_function_name
```

To run a basic csv write test in `write_csv.py`:
```bash
python write_csv.py csv_writer_test
```

## Tips to run specific crash file/files

Using the `pythonfuzz` decorator pass in `regression=True` with `dirs` having list of directories 
```python
@pythonfuzz(data_handle=CSVWriter, regression=True, dir=["/cudf/python/cudf/cudf/_fuzz_testing"])
```
