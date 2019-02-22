# Copyright (c) 2018, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import pytest
import os
import pandas as pd
import itertools


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
def json_files(request, tmp_path_factory, pdf):
    index, compression, orient = request.param
    if index is False and orient not in ('split', 'table'):
        pytest.skip("'index=False' is only valid when 'orient' is 'split' or "
                    "'table'")
    if index is False and orient == 'table':
        pytest.skip("'index=False' isn't valid when 'orient' is 'table'")
    fname_df = tmp_path_factory.mktemp("json") / "test_df.json"
    fname_series = tmp_path_factory.mktemp("json") / "test_series.json"
    pdf.to_json(fname_df, index=index, compression=compression,
                orient=orient)
    pdf['col_int32'].to_json(fname_series, index=index,
                             compression=compression, orient=orient)
    return (fname_df, fname_series, orient, compression)


@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_reader(json_files):
    path_df, path_series, orient, compression = json_files
    expect_df = pd.read_json(path_df, orient=orient, compression=compression)
    got_df = cudf.read_json(path_df, orient=orient, compression=compression)
    if len(expect_df) == 0:
        expect_df = expect_df.reset_index(drop=True)
        expect_df.columns = expect_df.columns.astype('object')
    if len(got_df) == 0:
        got_df = got_df.reset_index(drop=True)

    assert_eq(expect_df, got_df, check_categorical=False)

    # Only these orients are allowed for Series, but isn't enforced by Pandas
    if orient in ('split', 'records', 'index'):
        expect_series = pd.read_json(
            path_series, orient=orient, compression=compression, typ='series'
        )
        got_series = cudf.read_json(
            path_series, orient=orient, compression=compression, typ='series'
        )
        if len(expect_series) == 0:
            expect_series = expect_series.reset_index(drop=True)
        if len(got_df) == 0:
            got_series = got_series.reset_index(drop=True)

        assert_eq(expect_series, got_series)


@pytest.mark.filterwarnings("ignore:Can't infer compression")
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_json_writer(tmpdir, pdf, gdf):
    pdf_df_fname = tmpdir.join("pdf_df.json")
    gdf_df_fname = tmpdir.join("gdf_df.json")

    pdf.to_json(pdf_df_fname)
    gdf.to_json(gdf_df_fname)

    assert(os.path.exists(pdf_df_fname))
    assert(os.path.exists(gdf_df_fname))

    expect_df = pd.read_json(pdf_df_fname)
    got_df = pd.read_json(gdf_df_fname)

    assert_eq(expect_df, got_df)

    for column in pdf.columns:
        pdf_series_fname = tmpdir.join(column + "_" + "pdf_series.json")
        gdf_series_fname = tmpdir.join(column + "_" + "gdf_series.json")

        pdf[column].to_json(pdf_series_fname)
        gdf[column].to_json(gdf_series_fname)

        assert(os.path.exists(pdf_series_fname))
        assert(os.path.exists(gdf_series_fname))

        expect_series = pd.read_json(pdf_series_fname, typ='series')
        got_series = pd.read_json(gdf_series_fname, typ='series')

        assert_eq(expect_series, got_series)
