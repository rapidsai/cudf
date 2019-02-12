# Copyright (c) 2018, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import pytest
import os
import pandas as pd
import itertools


@pytest.fixture(params=[0, 1, 10, 100])
def pdf(request):
    print("nrows: " + str(request.param))
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

    return test_pdf


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


index_params = [True, False]
compression_params = ['gzip', 'bz2', 'zip', 'xz', None]
orient_params = ['columns', 'records', 'table', 'split']
params = itertools.product(
    index_params,
    compression_params,
    orient_params
)


@pytest.fixture(params=params)
def json_file(request, tmp_path_factory, pdf):
    index, compression, orient = request.param
    if index is False and orient not in ('split', 'table'):
        pytest.skip("'index=False' is only valid when 'orient' is 'split' or "
                    "'table'")
    if index is False and orient == 'table':
        pytest.skip("'index=False' isn't valid when 'orient' is 'table'")
    print(index, compression, orient)
    fname = tmp_path_factory.mktemp("json") / "test.json"
    pdf.to_json(fname, index=index, compression=compression,
                orient=orient)
    return (fname, orient, compression)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_reader(json_file):
    path, orient, compression = json_file
    expect = pd.read_json(path, orient=orient, compression=compression)
    got = cudf.read_json(path, orient=orient, compression=compression)
    if len(expect) == 0:
        expect = expect.reset_index(drop=True)
        expect.columns = expect.columns.astype('object')
    if len(got) == 0:
        got = got.reset_index()

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.filterwarnings("ignore:Can't infer compression")
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_writer(tmpdir, pdf, gdf):
    pdf_fname = tmpdir.join("pdf.json")
    gdf_fname = tmpdir.join("gdf.json")

    pdf.to_json(pdf_fname)
    gdf.to_json(gdf_fname)

    assert(os.path.exists(pdf_fname))
    assert(os.path.exists(gdf_fname))

    expect = pd.read_json(pdf_fname)
    got = pd.read_json(gdf_fname)

    assert_eq(expect, got)
