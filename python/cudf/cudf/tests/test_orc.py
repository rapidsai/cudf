# Copyright (c) 2019, NVIDIA CORPORATION.

import os
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "orc"


@pytest.fixture
def path_or_buf(datadir):
    fname = datadir / "TestOrcFile.test1.orc"
    try:
        with open(fname, "rb") as f:
            buffer = BytesIO(f.read())
    except Exception as excpr:
        if type(excpr).__name__ == "FileNotFoundError":
            pytest.skip(".parquet file is not found")
        else:
            print(type(excpr).__name__)

    def _make_path_or_buf(src):
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

    yield _make_path_or_buf


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize("engine", ["pyarrow", "cudf"])
@pytest.mark.parametrize("use_index", [False, True])
@pytest.mark.parametrize(
    "inputfile, columns",
    [
        ("TestOrcFile.emptyFile.orc", ["boolean1"]),
        (
            "TestOrcFile.test1.orc",
            [
                "boolean1",
                "byte1",
                "short1",
                "int1",
                "long1",
                "float1",
                "double1",
            ],
        ),
        ("TestOrcFile.RLEv2.orc", ["x", "y"]),
        ("TestOrcFile.testSnappy.orc", None),
        ("TestOrcFile.demo-12-zlib.orc", ["_col2", "_col3", "_col4", "_col5"]),
    ],
)
def test_orc_reader_basic(datadir, inputfile, columns, use_index, engine):
    path = datadir / inputfile
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    expect = orcfile.read(columns=columns).to_pandas()
    got = cudf.read_orc(
        path, engine=engine, columns=columns, use_index=use_index
    )

    assert_eq(expect, got, check_categorical=False)


def test_orc_reader_decimal(datadir):
    path = datadir / "TestOrcFile.decimal.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(path, engine="cudf").to_pandas()

    # Convert the decimal dtype from PyArrow to float64 for comparison to cuDF
    # This is because cuDF returns as float64 as it lacks an equivalent dtype
    pdf = pdf.apply(pd.to_numeric)

    np.testing.assert_allclose(pdf, gdf)


def test_orc_reader_decimal_as_int(datadir):
    path = datadir / "TestOrcFile.decimal.orc"

    gdf = cudf.read_orc(
        path, engine="cudf", decimals_as_float=False, force_decimal_scale=2
    ).to_pandas()

    assert gdf["_col0"][0] == -100050  # -1000.5


def test_orc_reader_filenotfound(tmpdir):
    with pytest.raises(FileNotFoundError):
        cudf.read_orc("TestMissingFile.orc")

    with pytest.raises(FileNotFoundError):
        cudf.read_orc(tmpdir.mkdir("cudf_orc"))


def test_orc_reader_local_filepath():
    path = "~/TestLocalFile.orc"
    if not os.path.isfile(path):
        pytest.skip("Local .orc file is not found")

    cudf.read_orc(path)


@pytest.mark.parametrize(
    "src", ["filepath", "pathobj", "bytes_io", "bytes", "url"]
)
def test_orc_reader_filepath_or_buffer(path_or_buf, src):
    cols = ["int1", "long1", "float1", "double1"]

    orcfile = pa.orc.ORCFile(path_or_buf("filepath"))
    expect = orcfile.read(columns=cols).to_pandas()
    got = cudf.read_orc(path_or_buf(src), columns=cols)

    assert_eq(expect, got)


def test_orc_reader_trailing_nulls(datadir):
    path = datadir / "TestOrcFile.nulls-at-end-snappy.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    expect = orcfile.read().to_pandas().fillna(0)
    got = cudf.read_orc(path, engine="cudf").fillna(0)

    # PANDAS uses NaN to represent invalid data, which forces float dtype
    # For comparison, we can replace NaN with 0 and cast to the cuDF dtype
    for col in expect.columns:
        expect[col] = expect[col].astype(got[col].dtype)

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.parametrize("use_index", [False, True])
@pytest.mark.parametrize(
    "inputfile",
    ["TestOrcFile.testDate1900.orc", "TestOrcFile.testDate2038.orc"],
)
def test_orc_reader_datetimestamp(datadir, inputfile, use_index):
    path = datadir / inputfile
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas(date_as_object=False)
    gdf = cudf.read_orc(path, engine="cudf", use_index=use_index)

    assert_eq(pdf, gdf, check_categorical=False)


def test_orc_reader_strings(datadir):
    path = datadir / "TestOrcFile.testStringAndBinaryStatistics.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    expect = orcfile.read(columns=["string1"])
    got = cudf.read_orc(path, engine="cudf", columns=["string1"])

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.parametrize("engine", ["cudf", "pyarrow"])
def test_orc_read_stripes(datadir, engine):
    path = datadir / "TestOrcFile.testDate1900.orc"
    try:
        pdf = cudf.read_orc(path, engine=engine)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    num_rows, stripes, col_names = cudf.io.read_orc_metadata(path)

    # Read stripes one at a time
    gdf = [
        cudf.read_orc(path, engine=engine, stripes=[i]) for i in range(stripes)
    ]
    gdf = cudf.concat(gdf).reset_index(drop=True)
    assert_eq(pdf, gdf, check_categorical=False)

    # Read stripes all at once
    gdf = cudf.read_orc(path, engine=engine, stripes=range(stripes))
    assert_eq(pdf, gdf, check_categorical=False)

    # Read only some stripes
    gdf = cudf.read_orc(path, engine=engine, stripes=[0, 1])
    assert_eq(gdf, pdf.head(25000))
    gdf = cudf.read_orc(path, engine=engine, stripes=[0, stripes - 1])
    assert_eq(
        gdf, cudf.concat([pdf.head(15000), pdf.tail(10000)], ignore_index=True)
    )


@pytest.mark.parametrize("num_rows", [1, 100, 3000])
@pytest.mark.parametrize("skiprows", [0, 1, 3000])
def test_orc_read_rows(datadir, skiprows, num_rows):
    path = datadir / "TestOrcFile.decimal.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(
        path, engine="cudf", skiprows=skiprows, num_rows=num_rows
    ).to_pandas()

    # Convert the decimal dtype from PyArrow to float64 for comparison to cuDF
    # This is because cuDF returns as float64 as it lacks an equivalent dtype
    pdf = pdf.apply(pd.to_numeric)

    # Slice rows out of the whole dataframe for comparison as PyArrow doesn't
    # have an API to read a subsection of rows from the file
    pdf = pdf[skiprows : skiprows + num_rows]

    np.testing.assert_allclose(pdf, gdf)


def test_orc_reader_uncompressed_block(datadir):
    path = datadir / "uncompressed_snappy.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    expect = orcfile.read().to_pandas()
    got = cudf.read_orc(path, engine="cudf")

    assert_eq(expect, got, check_categorical=False)


def test_orc_reader_nodata_block(datadir):
    path = datadir / "nodata.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except Exception as excpr:
        if type(excpr).__name__ == "ArrowIOError":
            pytest.skip(".orc file is not found")
        else:
            print(type(excpr).__name__)

    expect = orcfile.read().to_pandas()
    got = cudf.read_orc(path, engine="cudf", num_rows=1)

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.parametrize("compression", [None, "snappy"])
@pytest.mark.parametrize(
    "reference_file, columns",
    [
        (
            "TestOrcFile.test1.orc",
            [
                "boolean1",
                "byte1",
                "short1",
                "int1",
                "long1",
                "float1",
                "double1",
            ],
        ),
        ("TestOrcFile.demo-12-zlib.orc", ["_col1", "_col3", "_col5"]),
    ],
)
def test_orc_writer(datadir, tmpdir, reference_file, columns, compression):
    pdf_fname = datadir / reference_file
    gdf_fname = tmpdir.join("gdf.orc")

    try:
        orcfile = pa.orc.ORCFile(pdf_fname)
    except Exception as excpr:
        if type(excpr).__name__ == "ArrowIOError":
            pytest.skip(".orc file is not found")
        else:
            print(type(excpr).__name__)

    expect = orcfile.read(columns=columns).to_pandas()
    cudf.from_pandas(expect).to_orc(gdf_fname.strpath, compression=compression)
    got = pa.orc.ORCFile(gdf_fname).read(columns=columns).to_pandas()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtypes",
    [
        {"c": str, "a": int},
        {"c": int, "a": str},
        {"c": int, "a": str, "b": float},
        {"c": str, "a": object},
    ],
)
def test_orc_writer_strings(tmpdir, dtypes):
    gdf_fname = tmpdir.join("gdf_strings.orc")

    expect = cudf.datasets.randomdata(nrows=10, dtypes=dtypes, seed=1)
    expect.to_orc(gdf_fname)
    got = pa.orc.ORCFile(gdf_fname).read().to_pandas()

    assert_eq(expect, got)


def test_orc_writer_sliced(tmpdir):
    cudf_path = tmpdir.join("cudf.orc")

    df = pd.DataFrame()
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df = cudf.from_pandas(df)

    df_select = df.iloc[1:3]

    df_select.to_orc(cudf_path)
    assert_eq(cudf.read_orc(cudf_path), df_select.reset_index(drop=True))


@pytest.mark.parametrize(
    "orc_file",
    [
        "TestOrcFile.decimal.same.values.orc",
        "TestOrcFile.decimal.multiple.values.orc",
    ],
)
def test_orc_reader_decimal_type(datadir, orc_file):
    file_path = datadir / orc_file
    pdf = pd.read_orc(file_path)
    df = cudf.read_orc(file_path).to_pandas()
    # Converting to strings since pandas keeps it in decimal
    pdf["col8"] = pdf["col8"].astype("str")
    df["col8"] = df["col8"].astype("str")

    assert_eq(pdf, df)
