# Copyright (c) 2018, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import pytest
import os
import pandas as pd
import numpy as np
import pyarrow as pa
from string import ascii_letters


@pytest.fixture(params=[0, 1, 10, 100])
def pdf(request):
    types = ['bool', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64',
             'datetime64[ms]']
    renamer = {'C_l0_g' + str(idx): 'col_' + val for (idx, val) in
               enumerate(types)}
    typer = {'col_' + val: val for val in types}
    ncols = len(types)
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.util.testing.makeCustomDataframe(
        nrows=nrows,
        ncols=ncols,
        data_gen_f=lambda r, c: r,
        r_idx_type='i'
    )
    # Delete the name of the column index, and rename the row index
    del(test_pdf.columns.name)
    test_pdf.index.name = 'test_index'

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.astype('object')\
                       .rename(renamer, axis=1)\
                       .astype(typer)

    # Create non-numeric categorical data otherwise parquet may typecast it
    data = [ascii_letters[np.random.randint(0, 52)] for i in
            range(nrows)]
    test_pdf['col_category'] = pd.Series(data, dtype='category')

    return test_pdf


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


@pytest.fixture(params=['snappy', 'gzip', 'brotli', None])
def parquet_file(request, tmp_path_factory, pdf):
    fname = tmp_path_factory.mktemp("parquet") / "test.parquet"
    pdf.to_parquet(fname, engine='pyarrow', compression=request.param)
    return fname


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize('columns', [['col_int8'], ['col_category'],
                                     ['col_int32', 'col_float32'], None])
def test_parquet_reader(parquet_file, columns):
    expect = pd.read_parquet(parquet_file, columns=columns)
    got = cudf.read_parquet(parquet_file, columns=columns)
    if len(expect) == 0:
        expect = expect.reset_index(drop=True)

    # Pandas parquet reader converts the Arrow Dictionary Array to an `object`
    # dtype instead of a `categorical` dtype even though the metadata
    # explicitly states pandas_dtype `categorical`
    if 'col_category' in expect.columns:
        expect['col_category'] = expect['col_category'].astype('category')

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_parquet_writer(tmpdir, pdf, gdf):
    pdf_fname = tmpdir.join("pdf.parquet")
    gdf_fname = tmpdir.join("gdf.parquet")

    pdf.to_parquet(pdf_fname)
    gdf.to_parquet(gdf_fname)

    assert(os.path.exists(pdf_fname))
    assert(os.path.exists(gdf_fname))

    expect = pa.parquet.read_pandas(pdf_fname)
    got = pa.parquet.read_pandas(gdf_fname)

    # Pandas uses a datetime64[ns] while we use a datetime64[ms]
    expect_idx = expect.schema.get_field_index('col_datetime64[ms]')
    got_idx = got.schema.get_field_index('col_datetime64[ms]')
    expect = expect.set_column(
        expect_idx,
        expect.column(expect_idx).cast(pa.date64())
    )
    expect = expect.replace_schema_metadata()
    got = got.set_column(
        got_idx,
        got.column(got_idx).cast(pa.date64())
    )
    got = got.replace_schema_metadata()

    # assert_eq(expect, got)
    assert pa.Table.equals(expect, got)
