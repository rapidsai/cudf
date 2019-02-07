# Copyright (c) 2018, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import pytest
import os
import pandas as pd


@pytest.fixture
def pdf():
    return pd.DataFrame({'x': range(10),
                         'y': range(10)})


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


@pytest.fixture(params=['snappy', 'gzip', 'botli', None])
def parquet_file(request, tmp_path_factory, pdf):
    fname = tmp_path_factory.mktemp("parquet").join("test.parquet")
    pdf.to_parquet(fname, engine='pyarrow', compression=request.param)
    return fname


@pytest.mark.parametrize('columns', [['x'], ['y'], ['x', 'y'], None])
def test_parquet_reader(parquet_file, columns):
    expect = pd.read_parquet(parquet_file, columns=columns)
    got = cudf.read_parquet(parquet_file, columns=columns)
    assert_eq(expect, got)


def test_parquet_writer(tmpdir, pdf, gdf):
    pdf_fname = tmpdir.join("pdf.parquet")
    gdf_fname = tmpdir.join("gdf.parquet")

    pdf.to_parquet(pdf_fname)
    gdf.to_parquet(gdf_fname)

    assert(os.path.exists(pdf_fname))
    assert(os.path.exists(gdf_fname))

    expect = pd.read_parquet(pdf_fname)
    got = pd.read_parquet(gdf_fname)

    assert_eq(expect, got)
