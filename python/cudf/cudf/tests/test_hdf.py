# Copyright (c) 2018, NVIDIA CORPORATION.

import os
from string import ascii_letters

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import DATETIME_TYPES, NUMERIC_TYPES, assert_eq

try:
    import tables  # noqa F401
except ImportError:
    pytest.skip(
        "PyTables is not installed and is required for HDF " "reading/writing",
        allow_module_level=True,
    )


@pytest.fixture(params=[0, 1, 10, 100])
def pdf(request):
    types = NUMERIC_TYPES + DATETIME_TYPES + ["bool"]
    renamer = {
        "C_l0_g" + str(idx): "col_" + val for (idx, val) in enumerate(types)
    }
    typer = {"col_" + val: val for val in types}
    ncols = len(types)
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd._testing.makeCustomDataframe(
        nrows=nrows, ncols=ncols, data_gen_f=lambda r, c: r, r_idx_type="i"
    )
    # Delete the name of the column index, and rename the row index
    del test_pdf.columns.name
    test_pdf.index.name = "test_index"

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = (
        test_pdf.astype("object")
        .rename(renamer, axis=1)
        .astype(typer)
        .rename({"col_datetime64[ms]": "col_datetime64"}, axis=1)
    )

    # Create non-numeric categorical data otherwise may be typecasted
    data = [ascii_letters[np.random.randint(0, 52)] for i in range(nrows)]
    test_pdf["col_category"] = pd.Series(data, dtype="category")

    return (test_pdf, nrows)


@pytest.fixture
def gdf(pdf):
    pdf, nrows = pdf
    return (cudf.DataFrame.from_pandas(pdf), nrows)


@pytest.fixture(params=["fixed", "table"])
def hdf_files(request, tmp_path_factory, pdf):
    pdf, nrows = pdf
    if request.param == "fixed":
        pdf = pdf.drop("col_category", axis=1)

    fname_df = tmp_path_factory.mktemp("hdf") / "test_df.hdf"
    pdf.to_hdf(fname_df, "hdf_df_tests", format=request.param)

    fname_series = {}
    for column in pdf.columns:
        fname_series[column] = (
            tmp_path_factory.mktemp("hdf") / "test_series.hdf"
        )
        pdf[column].to_hdf(
            fname_series[column], "hdf_series_tests", format=request.param
        )
    return (fname_df, fname_series, request.param, nrows)


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize(
    "columns",
    [["col_int8"], ["col_category"], ["col_int32", "col_float32"], None],
)
def test_hdf_reader(hdf_files, columns):
    hdf_df_file, hdf_series, format, nrows = hdf_files
    if format == "fixed" and columns is not None:
        pytest.skip("Can't use columns with format 'fixed'")
    if format == "table" and nrows == 0:
        pytest.skip("Can't read 0 row table with format 'table'")
    expect_df = pd.read_hdf(hdf_df_file, columns=columns)
    got_df = cudf.read_hdf(hdf_df_file, columns=columns)

    assert_eq(expect_df, got_df, check_categorical=False)

    for column in hdf_series.keys():

        expect_series = pd.read_hdf(hdf_series[column])
        got_series = cudf.read_hdf(hdf_series[column])

        assert_eq(expect_series, got_series)


@pytest.mark.parametrize("format", ["fixed", "table"])
@pytest.mark.parametrize("complib", ["zlib", "bzip2", "lzo", "blosc"])
@pytest.mark.filterwarnings("ignore:Using CPU")
def test_hdf_writer(tmpdir, pdf, gdf, complib, format):
    pdf, nrows = pdf
    gdf, _ = gdf

    if format == "fixed":
        pdf = pdf.drop("col_category", axis=1)
        gdf = gdf.drop("col_category", axis=1)

    pdf_df_fname = tmpdir.join("pdf_df.hdf")
    gdf_df_fname = tmpdir.join("gdf_df.hdf")

    pdf.to_hdf(pdf_df_fname, "hdf_tests", format=format, complib=complib)
    gdf.to_hdf(gdf_df_fname, "hdf_tests", format=format, complib=complib)

    assert os.path.exists(pdf_df_fname)
    assert os.path.exists(gdf_df_fname)

    if format == "table" and nrows == 0:
        pytest.skip("Can't read 0 row table with format 'table'")

    expect = pd.read_hdf(pdf_df_fname)
    got = pd.read_hdf(gdf_df_fname)

    assert_eq(expect, got)

    for column in pdf.columns:
        pdf_series_fname = tmpdir.join(column + "_" + "pdf_series.hdf")
        gdf_series_fname = tmpdir.join(column + "_" + "gdf_series.hdf")

        pdf[column].to_hdf(
            pdf_series_fname, "hdf_tests", format=format, complib=complib
        )
        gdf[column].to_hdf(
            gdf_series_fname, "hdf_tests", format=format, complib=complib
        )

        assert os.path.exists(pdf_series_fname)
        assert os.path.exists(gdf_series_fname)

        expect_series = pd.read_hdf(pdf_series_fname)
        got_series = pd.read_hdf(gdf_series_fname)

        assert_eq(expect_series, got_series)
