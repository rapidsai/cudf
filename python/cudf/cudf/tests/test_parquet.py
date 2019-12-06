# Copyright (c) 2019, NVIDIA CORPORATION.

import os
import random
from glob import glob
from io import BytesIO
from string import ascii_letters

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "parquet"


@pytest.fixture(params=[0, 1, 10, 100])
def pdf(request):
    types = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[ms]",
    ]
    renamer = {
        "C_l0_g" + str(idx): "col_" + val for (idx, val) in enumerate(types)
    }
    typer = {"col_" + val: val for val in types}
    ncols = len(types)
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd.util.testing.makeCustomDataframe(
        nrows=nrows, ncols=ncols, data_gen_f=lambda r, c: r, r_idx_type="i"
    )
    # Delete the name of the column index, and rename the row index
    del test_pdf.columns.name
    test_pdf.index.name = "test_index"

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.astype("object").rename(renamer, axis=1).astype(typer)

    # Create non-numeric categorical data otherwise parquet may typecast it
    data = [ascii_letters[np.random.randint(0, 52)] for i in range(nrows)]
    test_pdf["col_category"] = pd.Series(data, dtype="category")

    return test_pdf


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


@pytest.fixture(params=["snappy", "gzip", "brotli", None])
def parquet_file(request, tmp_path_factory, pdf):
    fname = tmp_path_factory.mktemp("parquet") / "test.parquet"
    pdf.to_parquet(fname, engine="pyarrow", compression=request.param)
    return fname


def make_pdf(nrows, ncolumns=1, nvalids=0, dtype=np.int64):
    test_pdf = pd.util.testing.makeCustomDataframe(
        nrows=nrows,
        ncols=1,
        data_gen_f=lambda r, c: r,
        dtype=dtype,
        r_idx_type="i",
    )
    del test_pdf.columns.name

    # Randomly but reproducibly mark subset of rows as invalid
    random.seed(1337)
    mask = random.sample(range(nrows), nvalids)
    test_pdf[test_pdf.index.isin(mask)] = np.NaN

    return test_pdf


@pytest.fixture
def parquet_path_or_buf(datadir):
    fname = datadir / "spark_timestamp.snappy.parquet"
    try:
        with open(fname, "rb") as f:
            buffer = BytesIO(f.read())
    except Exception as excpr:
        if type(excpr).__name__ == "FileNotFoundError":
            pytest.skip(".parquet file is not found")
        else:
            print(type(excpr).__name__)

    def _make_parquet_path_or_buf(src):
        if src == "filepath":
            return str(fname)
        if src == "pathobj":
            return fname
        if src == "bytes_io":
            return buffer
        if src == "bytes":
            return buffer.getvalue()
        if src == "url":
            return fname.as_uri()

        raise ValueError("Invalid source type")

    yield _make_parquet_path_or_buf


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("engine", ["pyarrow", "cudf"])
@pytest.mark.parametrize(
    "columns",
    [
        ["col_int8"],
        ["col_category"],
        ["col_int32", "col_float32"],
        ["col_int16", "col_float64", "col_int8"],
        None,
    ],
)
def test_parquet_reader_basic(parquet_file, columns, engine):
    expect = pd.read_parquet(parquet_file, columns=columns)
    got = cudf.read_parquet(parquet_file, engine=engine, columns=columns)
    if len(expect) == 0:
        expect = expect.reset_index(drop=True)
        if "col_category" in expect.columns:
            expect["col_category"] = expect["col_category"].astype("category")

    # PANDAS returns category objects whereas cuDF returns hashes
    if engine == "cudf":
        if "col_category" in expect.columns:
            expect = expect.drop(columns=["col_category"])
        if "col_category" in got.columns:
            got = got.drop("col_category")

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.parametrize("has_null", [False, True])
@pytest.mark.parametrize("strings_to_categorical", [False, True, None])
def test_parquet_reader_strings(tmpdir, strings_to_categorical, has_null):
    df = pd.DataFrame(
        [(1, "aaa", 9.0), (2, "bbb", 8.0), (3, "ccc", 7.0)],
        columns=pd.Index(list("abc")),
    )
    if has_null:
        df.at[1, "b"] = None
    fname = tmpdir.join("test_pq_reader_strings.parquet")
    df.to_parquet(fname)
    assert os.path.exists(fname)

    if strings_to_categorical is not None:
        gdf = cudf.read_parquet(
            fname, engine="cudf", strings_to_categorical=strings_to_categorical
        )
    else:
        gdf = cudf.read_parquet(fname, engine="cudf")

    if strings_to_categorical:
        if has_null:
            hash_ref = [989983842, None, 1169108191]
        else:
            hash_ref = [989983842, 429364346, 1169108191]
        assert gdf["b"].dtype == np.dtype("int32")
        assert list(gdf["b"]) == list(hash_ref)
    else:
        assert gdf["b"].dtype == np.dtype("object")
        assert list(gdf["b"]) == list(df["b"])


@pytest.mark.parametrize("columns", [None, ["b"]])
@pytest.mark.parametrize("index_col", ["b", "Nameless", None])
def test_parquet_reader_index_col(tmpdir, index_col, columns):
    df = pd.DataFrame({"a": range(3), "b": range(3, 6), "c": range(6, 9)})

    if index_col is None:
        # No index column
        df.reset_index(drop=True, inplace=True)
    elif index_col == "Nameless":
        # Index column but no name
        df.set_index("a", inplace=True)
        df.index.name = None
    else:
        # Index column as normal
        df.set_index(index_col, inplace=True)

    fname = tmpdir.join("test_pq_reader_index_col.parquet")

    # PANDAS' PyArrow backend always writes the index unless disabled
    df.to_parquet(fname, index=(False if index_col is None else True))
    assert os.path.exists(fname)

    pdf = pd.read_parquet(fname, columns=columns)
    gdf = cudf.read_parquet(fname, engine="cudf", columns=columns)

    assert_eq(pdf, gdf, check_categorical=False)


@pytest.mark.parametrize("pandas_compat", [True, False])
@pytest.mark.parametrize("columns", [["a"], ["d"], ["a", "b"], None])
def test_parquet_reader_pandas_metadata(tmpdir, columns, pandas_compat):
    df = pd.DataFrame({"a": range(6, 9), "b": range(3, 6), "c": range(6, 9)})
    df.set_index("b", inplace=True)

    fname = tmpdir.join("test_pq_reader_pandas_metadata.parquet")
    df.to_parquet(fname)
    assert os.path.exists(fname)

    # PANDAS `read_parquet()` and PyArrow `read_pandas()` always includes index
    # Instead, directly use PyArrow to optionally omit the index
    expect = pa.parquet.read_table(
        fname, columns=columns, use_pandas_metadata=pandas_compat
    ).to_pandas()
    got = cudf.read_parquet(
        fname, columns=columns, use_pandas_metadata=pandas_compat
    )

    if pandas_compat or columns is None or "b" in columns:
        assert got.index.name == "b"
    else:
        assert got.index.name is None
    assert_eq(expect, got, check_categorical=False)


def test_parquet_read_metadata(tmpdir, pdf):
    def num_row_groups(rows, group_size):
        return max(1, (rows + (group_size - 1)) // group_size)

    fname = tmpdir.join("metadata.parquet")
    row_group_size = 5
    pdf.to_parquet(fname, compression="snappy", row_group_size=row_group_size)

    num_rows, row_groups, col_names = cudf.io.read_parquet_metadata(fname)

    assert num_rows == len(pdf.index)
    assert row_groups == num_row_groups(num_rows, row_group_size)
    for a, b in zip(col_names, pdf.columns):
        assert a == b


@pytest.mark.parametrize("row_group_size", [1, 5, 100])
def test_parquet_read_row_group(tmpdir, pdf, row_group_size):
    fname = tmpdir.join("row_group.parquet")
    pdf.to_parquet(fname, compression="gzip", row_group_size=row_group_size)

    num_rows, row_groups, col_names = cudf.io.read_parquet_metadata(fname)

    gdf = [cudf.read_parquet(fname, row_group=i) for i in range(row_groups)]
    gdf = cudf.concat(gdf).reset_index(drop=True)

    if "col_category" in pdf.columns:
        pdf = pdf.drop(columns=["col_category"])
    if "col_category" in gdf.columns:
        gdf = gdf.drop("col_category")

    assert_eq(pdf.reset_index(drop=True), gdf, check_categorical=False)


@pytest.mark.parametrize("row_group_size", [1, 4, 33])
def test_parquet_read_rows(tmpdir, pdf, row_group_size):
    fname = tmpdir.join("row_group.parquet")
    pdf.to_parquet(fname, compression="None", row_group_size=row_group_size)

    total_rows, row_groups, col_names = cudf.io.read_parquet_metadata(fname)

    num_rows = total_rows // 4
    skip_rows = (total_rows - num_rows) // 2
    gdf = cudf.read_parquet(fname, skip_rows=skip_rows, num_rows=num_rows)

    if "col_category" in pdf.columns:
        pdf = pdf.drop(columns=["col_category"])
    if "col_category" in gdf.columns:
        gdf = gdf.drop("col_category")

    for row in range(num_rows):
        assert gdf["col_int32"][row] == row + skip_rows


def test_parquet_reader_spark_timestamps(datadir):
    fname = datadir / "spark_timestamp.snappy.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_spark_decimals(datadir):
    fname = datadir / "spark_decimal.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    # Convert the decimal dtype from PyArrow to float64 for comparison to cuDF
    # This is because cuDF returns as float64 as it lacks an equivalent dtype
    expect = expect.apply(pd.to_numeric)

    # np.testing.assert_allclose(expect, got)
    assert_eq(expect, got)


def test_parquet_reader_microsecond_timestamps(datadir):
    fname = datadir / "usec_timestamp.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_mixedcompression(datadir):
    fname = datadir / "mixed_compression.parquet"

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_reader_invalids(tmpdir):
    test_pdf = make_pdf(nrows=1000, nvalids=1000 // 4, dtype=np.int64)

    fname = tmpdir.join("invalids.parquet")
    test_pdf.to_parquet(fname, engine="pyarrow")

    expect = pd.read_parquet(fname)
    got = cudf.read_parquet(fname)

    assert_eq(expect, got)


def test_parquet_chunked_skiprows(tmpdir):
    processed = 0
    batch = 10000
    n = 100000
    out_df = cudf.DataFrame(
        {
            "y": np.arange(n),
            "z": np.random.choice(range(1000000, 2000000), n, replace=False),
            "s": np.random.choice(range(20), n, replace=True),
            "a": np.round(np.random.uniform(1, 5000, n), 2),
        }
    )

    fname = tmpdir.join("skiprows.parquet")
    out_df.to_pandas().to_parquet(fname)

    for i in range(10):
        chunk = cudf.read_parquet(fname, skip_rows=processed, num_rows=batch)
        expect = out_df[processed : processed + batch].reset_index(drop=True)
        assert_eq(chunk.reset_index(drop=True), expect)
        processed += batch
        del chunk


def test_parquet_reader_filenotfound(tmpdir):
    with pytest.raises(FileNotFoundError):
        cudf.read_parquet("TestMissingFile.parquet")

    with pytest.raises(FileNotFoundError):
        cudf.read_parquet(tmpdir.mkdir("cudf_parquet"))


def test_parquet_reader_local_filepath():
    fname = "~/TestLocalFile.parquet"
    if not os.path.isfile(fname):
        pytest.skip("Local .parquet file is not found")

    cudf.read_parquet(fname)


@pytest.mark.parametrize(
    "src", ["filepath", "pathobj", "bytes_io", "bytes", "url"]
)
def test_parquet_reader_filepath_or_buffer(parquet_path_or_buf, src):
    expect = pd.read_parquet(parquet_path_or_buf("filepath"))
    got = cudf.read_parquet(parquet_path_or_buf(src))

    assert_eq(expect, got)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_parquet_writer(tmpdir, pdf, gdf):
    pdf_fname = tmpdir.join("pdf.parquet")
    gdf_fname = tmpdir.join("gdf.parquet")

    pdf.to_parquet(pdf_fname.strpath)
    gdf.to_parquet(gdf_fname.strpath)

    assert os.path.exists(pdf_fname)
    assert os.path.exists(gdf_fname)

    expect = pa.parquet.read_pandas(pdf_fname)
    got = pa.parquet.read_pandas(gdf_fname)

    def clone_field(table, name, datatype):
        f = table.schema.field_by_name(name)
        return pa.field(f.name, datatype, f.nullable, f.metadata)

    # Pandas uses a datetime64[ns] while we use a datetime64[ms]
    expect_idx = expect.schema.get_field_index("col_datetime64[ms]")
    expect_field = clone_field(expect, "col_datetime64[ms]", pa.date64())
    expect = expect.set_column(
        expect_idx,
        expect_field,
        expect.column(expect_idx).cast(expect_field.type),
    )
    expect = expect.replace_schema_metadata()

    got_idx = got.schema.get_field_index("col_datetime64[ms]")
    got_field = clone_field(got, "col_datetime64[ms]", pa.date64())
    got = got.set_column(
        got_idx, got_field, got.column(got_idx).cast(got_field.type)
    )
    got = got.replace_schema_metadata()

    # assert_eq(expect, got)
    assert pa.Table.equals(expect, got)


def test_multifile_warning(datadir):
    fpath = datadir.__fspath__() + "/*.parquet"
    with pytest.warns(UserWarning):
        got = cudf.read_parquet(fpath)
        fname = sorted(glob(fpath))[0]
        expect = pd.read_parquet(fname)
        expect = expect.apply(pd.to_numeric)
        assert_eq(expect, got)
