# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import os
from string import ascii_letters

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import NUMERIC_TYPES


@pytest.fixture(params=[0, 1, 10, 100])
def pdf(request):
    rng = np.random.default_rng(seed=0)
    types = [*NUMERIC_TYPES, "bool"]
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.DataFrame(
        {
            f"col_{typ}": rng.integers(0, nrows, nrows).astype(typ)
            for typ in types
        }
    )
    # Delete the name of the column index, and rename the row index
    test_pdf.columns.name = None
    test_pdf.index.name = "index"

    # Create non-numeric categorical data otherwise may get typecasted
    data = [ascii_letters[rng.integers(0, 52)] for i in range(nrows)]
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
