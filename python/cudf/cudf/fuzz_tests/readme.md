# Fuzz Tests

This directory contains all the Fuzz tests for cudf library.


## Steps to write a fuzz test

1. Add a Data Handler class which actually generates the necessary random data according to your requirements. This class should be added in `cudf/cudf/testing/`. A sample data handler class is: `CSVWriter`: <URL>TODO</URL>
2. The above created Data Handler will have to be passed to the Fuzzer worker which continuously runs the fuzz tests. For this the method which has to be fuzz-tested should have the decorator by passing the Data Handler like:
  
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