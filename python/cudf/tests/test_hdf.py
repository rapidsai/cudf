# Copyright (c) 2018, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import pytest
import os
import pandas as pd
import numpy as np
from string import ascii_letters


try:
    import tables # noqa F401
    _have_pytables = True
except ImportError:
    _have_pytables = False
if not _have_pytables:
    pytest.skip('PyTables is not installed and is required for HDF '
                'reading/writing', allow_module_level=True)


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

    # Create non-numeric categorical data otherwise may be typecasted
    data = [ascii_letters[np.random.randint(0, 52)] for i in
            range(request.param)]
    test_pdf['col_category'] = pd.Series(data, dtype='category')

    return (test_pdf, nrows)


@pytest.fixture
def gdf(pdf):
    pdf, nrows = pdf
    return (cudf.DataFrame.from_pandas(pdf), nrows)


@pytest.fixture(params=['fixed', 'table'])
def hdf_file(request, tmp_path_factory, pdf):
    pdf, nrows = pdf
    if request.param == 'fixed':
        pdf = pdf.drop('col_category', axis=1)
    fname = tmp_path_factory.mktemp("hdf") / "test.hdf"
    pdf.to_hdf(fname, 'hdf_tests', format=request.param)
    return (fname, request.param, nrows)


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize('columns', [['col_int8'], ['col_category'],
                                     ['col_int32', 'col_float32'], None])
def test_hdf_reader(hdf_file, columns):
    hdf_file, format, nrows = hdf_file
    if format == 'fixed' and columns is not None:
        pytest.skip("Can't use columns with format 'fixed'")
    if format == 'table' and nrows == 0:
        pytest.skip("Can't read 0 row table with format 'table'")
    expect = pd.read_hdf(hdf_file, columns=columns)
    got = cudf.read_hdf(hdf_file, columns=columns)

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.parametrize('format', ['fixed', 'table'])
@pytest.mark.parametrize('complib', ['zlib', 'bzip2', 'lzo', 'blosc'])
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_hdf_writer(tmpdir, pdf, gdf, complib, format):
    pdf, nrows = pdf
    gdf, _ = gdf
    print(pdf.index.name, gdf.index.name)

    if format == 'fixed':
        pdf = pdf.drop('col_category', axis=1)
        gdf = gdf.drop('col_category')

    print(pdf.index.name, gdf.index.name)

    pdf_fname = tmpdir.join("pdf.hdf")
    gdf_fname = tmpdir.join("gdf.hdf")

    pdf.to_hdf(pdf_fname, 'hdf_tests', format=format, complib=complib)
    gdf.to_hdf(gdf_fname, 'hdf_tests', format=format, complib=complib)

    assert(os.path.exists(pdf_fname))
    assert(os.path.exists(gdf_fname))

    if format == 'table' and nrows == 0:
        pytest.skip("Can't read 0 row table with format 'table'")

    expect = pd.read_hdf(pdf_fname)
    got = pd.read_hdf(gdf_fname)

    assert_eq(expect, got)
