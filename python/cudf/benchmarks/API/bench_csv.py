import os

import pytest
import numpy as np

from utils import benchmark_with_object

import cudf
from cudf.config import set_config

@pytest.fixture
def tmp_file():
    handle = open("tmp.csv", "w")
    yield handle
    handle.close()
    os.remove("tmp.csv")

@pytest.fixture
def default_32_bits_integer():
    set_config("default_int_bitwidth", 32)
    yield
    set_config("default_int_bitwidth", 64)

@benchmark_with_object(cls="dataframe", dtype="int", nulls=False)
def bench_read_csv(benchmark, dataframe, tmp_file, default_32_bits_integer):
    dataframe.to_csv(tmp_file)
    benchmark(cudf.read_csv, "tmp.csv")
