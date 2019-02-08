# Copyright (c) 2018, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import pytest
import os
import pandas as pd
import numpy as np


@pytest.fixture
def pdf():
    types = ['bool', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64',
             'datetime64[ms]', 'category']
    renamer = {'C_l0_g' + str(idx): 'col_' + val for (idx, val) in
               enumerate(types)}
    typer = {'col_' + val: val for val in types}
    ncols = len(types)
    nrows = 10

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.util.testing.makeCustomDataframe(
        nrows=nrows,
        ncols=ncols,
        data_gen_f=lambda r, c: np.random.randint(1, 100),
        r_idx_type='i'
    )
    # Delete the name of the column index, and rename the row index
    del(test_pdf.columns.name)
    test_pdf.index.name = 'index'

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.astype('object')\
                       .rename(renamer, axis=1)\
                       .astype(typer)

    return test_pdf


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


@pytest.fixture(params=['snappy', 'gzip', 'botli', None])
def parquet_file(request, tmp_path_factory, pdf):
    fname = tmp_path_factory.mktemp("parquet") / "test.parquet"
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
