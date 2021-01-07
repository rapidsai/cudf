# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import os
import datetime
import math
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pytest

import cudf
from cudf.tests.utils import assert_eq, supported_numpy_dtypes, gen_rand_series


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


def test_orc_read_statistics(datadir):
    # Read in file containing 2 columns ("int1" and "string1") and 3 stripes
    # (sizes 5000, 5000 and 1000 respectively). Each stripe has the same value
    # in every one of its rows. The values the stripes have are 1, 2, and 3 in
    # "int1" and "one", "two", and "three" in "string1".
    path = datadir / "TestOrcFile.testStripeLevelStats.orc"
    try:
        (
            file_statistics,
            stripes_statistics,
        ) = cudf.io.orc.read_orc_statistics(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    # Check numberOfValues
    assert_eq(file_statistics["int1"]["number_of_values"], 11_000)
    assert_eq(
        file_statistics["int1"]["number_of_values"],
        sum(
            [
                stripes_statistics[0]["int1"]["number_of_values"],
                stripes_statistics[1]["int1"]["number_of_values"],
                stripes_statistics[2]["int1"]["number_of_values"],
            ]
        ),
    )
    assert_eq(
        stripes_statistics[1]["int1"]["number_of_values"],
        stripes_statistics[1]["string1"]["number_of_values"],
    )
    assert_eq(stripes_statistics[2]["string1"]["number_of_values"], 1_000)

    # Check other statistics
    assert_eq(stripes_statistics[2]["string1"]["has_null"], False)
    assert_eq(
        file_statistics["int1"]["minimum"],
        min(
            stripes_statistics[0]["int1"]["minimum"],
            stripes_statistics[1]["int1"]["minimum"],
            stripes_statistics[2]["int1"]["minimum"],
        ),
    )
    assert_eq(file_statistics["int1"]["minimum"], 1)
    assert_eq(file_statistics["string1"]["minimum"], "one")


@pytest.mark.parametrize("engine", ["cudf", "pyarrow"])
@pytest.mark.parametrize(
    "predicate,expected_len",
    [
        ([[("int1", "==", 1)]], 5000),
        ([[("int1", "<=", 2)]], 10000),
        ([[("int1", "==", -1)]], 0),
        ([[("int1", "in", range(3))]], 10000),
        ([[("int1", "in", {1, 3})]], 6000),
        ([[("int1", "not in", {1, 3})]], 5000),
    ],
)
def test_orc_read_filtered(datadir, engine, predicate, expected_len):
    path = datadir / "TestOrcFile.testStripeLevelStats.orc"
    try:
        df_filtered = cudf.read_orc(path, engine=engine, filters=predicate)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    # Assert # of rows after filtering
    assert len(df_filtered) == expected_len


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
        # For addional information take look at PR 7034
        "TestOrcFile.decimal.runpos.issue.orc",
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


# For addional information take look at PR 6636 and 6702
@pytest.mark.parametrize(
    "orc_file",
    [
        "TestOrcFile.boolean_corruption_PR_6636.orc",
        "TestOrcFile.boolean_corruption_PR_6702.orc",
    ],
)
def test_orc_reader_boolean_type(datadir, orc_file):
    file_path = datadir / orc_file

    pdf = pd.read_orc(file_path)
    df = cudf.read_orc(file_path).to_pandas()

    assert_eq(pdf, df)


@pytest.mark.filterwarnings("ignore:Using CPU")
def test_orc_reader_tzif_timestamps(datadir):
    # Contains timstamps in the range covered by the TZif file
    # Other timedate tests only cover "future" times
    path = datadir / "TestOrcFile.lima_timezone.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(path, engine="cudf").to_pandas()

    assert_eq(pdf, gdf)


def test_int_overflow(tmpdir):
    file_path = tmpdir.join("gdf_overflow.orc")

    # The number of rows and the large element trigger delta encoding
    num_rows = 513
    df = cudf.DataFrame({"a": [None] * num_rows}, dtype="int32")
    df["a"][0] = 1024 * 1024 * 1024
    df["a"][num_rows - 1] = 1
    df.to_orc(file_path)

    assert_eq(cudf.read_orc(file_path), df)


def normalized_equals(value1, value2):
    if isinstance(value1, (datetime.datetime, np.datetime64)):
        value1 = np.datetime64(value1, "ms")
    if isinstance(value2, (datetime.datetime, np.datetime64)):
        value2 = np.datetime64(value2, "ms")

    # Compare integers with floats now
    if isinstance(value1, float) or isinstance(value2, float):
        return math.isclose(value1, value2)

    return value1 == value2


@pytest.mark.parametrize("nrows", [1, 100, 6000000])
def test_orc_write_statistics(tmpdir, datadir, nrows):
    supported_stat_types = supported_numpy_dtypes + ["str"]

    # Make a dataframe
    gdf = cudf.DataFrame(
        {
            "col_" + str(dtype): gen_rand_series(dtype, nrows, has_nulls=True)
            for dtype in supported_stat_types
        }
    )
    fname = tmpdir.join("gdf.orc")

    # Write said dataframe to ORC with cuDF
    gdf.to_orc(fname.strpath)

    # Read back written ORC's statistics
    orc_file = pa.orc.ORCFile(fname)
    (file_stats, stripes_stats,) = cudf.io.orc.read_orc_statistics(fname)

    # check file stats
    for col in gdf:
        if "minimum" in file_stats[col]:
            stats_min = file_stats[col]["minimum"]
            actual_min = gdf[col].min()
            assert normalized_equals(actual_min, stats_min)
        if "maximum" in file_stats[col]:
            stats_max = file_stats[col]["maximum"]
            actual_max = gdf[col].max()
            assert normalized_equals(actual_max, stats_max)

    # compare stripe statistics with actual min/max
    for stripe_idx in range(0, orc_file.nstripes):
        stripe = orc_file.read_stripe(stripe_idx)
        # pandas is unable to handle min/max of string col with nulls
        stripe_df = cudf.DataFrame(stripe.to_pandas())
        for col in stripe_df:
            if "minimum" in stripes_stats[stripe_idx][col]:
                actual_min = stripe_df[col].min()
                stats_min = stripes_stats[stripe_idx][col]["minimum"]
                assert normalized_equals(actual_min, stats_min)

            if "maximum" in stripes_stats[stripe_idx][col]:
                actual_max = stripe_df[col].max()
                stats_max = stripes_stats[stripe_idx][col]["maximum"]
                assert normalized_equals(actual_max, stats_max)


@pytest.mark.parametrize("nrows", [1, 100, 6000000])
def test_orc_write_bool_statistics(tmpdir, datadir, nrows):
    # Make a dataframe
    gdf = cudf.DataFrame({"col_bool": gen_rand_series("bool", nrows)})
    fname = tmpdir.join("gdf.orc")

    # Write said dataframe to ORC with cuDF
    gdf.to_orc(fname.strpath)

    # Read back written ORC's statistics
    orc_file = pa.orc.ORCFile(fname)
    (file_stats, stripes_stats,) = cudf.io.orc.read_orc_statistics(fname)

    # check file stats
    col = "col_bool"
    if "true_count" in file_stats[col]:
        stats_true_count = file_stats[col]["true_count"]
        actual_true_count = gdf[col].sum()
        assert normalized_equals(actual_true_count, stats_true_count)

    if "number_of_values" in file_stats[col]:
        stats_valid_count = file_stats[col]["number_of_values"]
        actual_valid_count = gdf[col].valid_count
        assert normalized_equals(actual_valid_count, stats_valid_count)

    # compare stripe statistics with actual min/max
    for stripe_idx in range(0, orc_file.nstripes):
        stripe = orc_file.read_stripe(stripe_idx)
        # pandas is unable to handle min/max of string col with nulls
        stripe_df = cudf.DataFrame(stripe.to_pandas())

        if "true_count" in stripes_stats[stripe_idx][col]:
            actual_true_count = stripe_df[col].sum()
            stats_true_count = stripes_stats[stripe_idx][col]["true_count"]
            assert normalized_equals(actual_true_count, stats_true_count)

        if "number_of_values" in stripes_stats[stripe_idx][col]:
            actual_valid_count = stripe_df[col].valid_count
            stats_valid_count = stripes_stats[stripe_idx][col][
                "number_of_values"
            ]
            assert normalized_equals(actual_valid_count, stats_valid_count)


def test_orc_reader_gmt_timestamps(datadir):
    path = datadir / "TestOrcFile.gmt.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(path, engine="cudf").to_pandas()
    assert_eq(pdf, gdf)
