# Copyright (c) 2018, NVIDIA CORPORATION.

import os
from distutils.version import LooseVersion
from string import ascii_letters

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.tests.utils import NUMERIC_TYPES, assert_eq

if LooseVersion(pd.__version__) < LooseVersion("0.24"):
    try:
        import feather  # noqa F401
    except ImportError:
        pytest.skip(
            "Feather is not installed and is required for Pandas <" " 0.24",
            allow_module_level=True,
        )


@pytest.fixture(params=[0, 1, 10, 100])
def pdf(request):
    types = NUMERIC_TYPES + ["bool"]
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
    test_pdf.columns.name = None
    test_pdf.index.name = "index"

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.astype("object").rename(renamer, axis=1).astype(typer)

    # Create non-numeric categorical data otherwise may get typecasted
    data = [ascii_letters[np.random.randint(0, 52)] for i in range(nrows)]
    test_pdf["col_category"] = pd.Series(data, dtype="category")

    # Feather can't handle indexes properly
    test_pdf = test_pdf.reset_index(drop=True)
    test_pdf.index.name = None

    return test_pdf


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


@pytest.fixture
def feather_file(tmp_path_factory, pdf):
    fname = tmp_path_factory.mktemp("feather") / "test.feather"
    pdf.to_feather(fname)
    return fname


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize(
    "columns",
    [["col_int8"], ["col_category"], ["col_int32", "col_float32"], None],
)
def test_feather_reader(feather_file, columns):
    expect = pa.feather.read_table(feather_file, columns=columns).to_pandas()
    got = (
        cudf.read_feather(feather_file, columns=columns)
        .to_arrow(preserve_index=False)
        .to_pandas()
    )

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_feather_writer(tmpdir, pdf, gdf):
    pdf_fname = tmpdir.join("pdf.feather")
    gdf_fname = tmpdir.join("gdf.feather")

    pdf.to_feather(pdf_fname)
    gdf.to_feather(gdf_fname)

    assert os.path.exists(pdf_fname)
    assert os.path.exists(gdf_fname)

    expect = pa.feather.read_table(pdf_fname)
    got = pa.feather.read_table(gdf_fname)

    assert pa.Table.equals(expect, got)
