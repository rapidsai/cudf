# Copyright (c) 2019-2021, NVIDIA CORPORATION.

import datetime
import decimal
import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pyorc as po
import pytest

import cudf
from cudf.core.dtypes import Decimal64Dtype
from cudf.io.orc import ORCWriter
from cudf.testing._utils import (
    assert_eq,
    gen_rand_series,
    supported_numpy_dtypes,
)


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
        ) = cudf.io.orc.read_orc_statistics([path, path])
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    # Check numberOfValues
    assert_eq(file_statistics[0]["int1"]["number_of_values"], 11_000)
    assert_eq(
        file_statistics[0]["int1"]["number_of_values"],
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
        file_statistics[0]["int1"]["minimum"],
        min(
            stripes_statistics[0]["int1"]["minimum"],
            stripes_statistics[1]["int1"]["minimum"],
            stripes_statistics[2]["int1"]["minimum"],
        ),
    )
    assert_eq(file_statistics[0]["int1"]["minimum"], 1)
    assert_eq(file_statistics[0]["string1"]["minimum"], "one")


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
        cudf.read_orc(path, engine=engine, stripes=[[i]])
        for i in range(stripes)
    ]
    gdf = cudf.concat(gdf).reset_index(drop=True)
    assert_eq(pdf, gdf, check_categorical=False)

    # Read stripes all at once
    gdf = cudf.read_orc(
        path, engine=engine, stripes=[[int(x) for x in range(stripes)]]
    )
    assert_eq(pdf, gdf, check_categorical=False)

    # Read only some stripes
    gdf = cudf.read_orc(path, engine=engine, stripes=[[0, 1]])
    assert_eq(gdf, pdf.head(25000))
    gdf = cudf.read_orc(path, engine=engine, stripes=[[0, stripes - 1]])
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
    )

    # Slice rows out of the whole dataframe for comparison as PyArrow doesn't
    # have an API to read a subsection of rows from the file
    pdf = pdf[skiprows : skiprows + num_rows]
    pdf = pdf.reset_index(drop=True)

    assert_eq(pdf, gdf)


def test_orc_read_skiprows(tmpdir):
    buff = BytesIO()
    df = pd.DataFrame(
        {"a": [1, 0, 1, 0, None, 1, 1, 1, 0, None, 0, 0, 1, 1, 1, 1]},
        dtype=pd.BooleanDtype(),
    )
    writer = po.Writer(buff, po.Struct(a=po.Boolean()))
    tuples = list(
        map(
            lambda x: (None,) if x[0] is pd.NA else x,
            list(df.itertuples(index=False, name=None)),
        )
    )
    writer.writerows(tuples)
    writer.close()

    skiprows = 10

    expected = cudf.read_orc(buff)[skiprows::].reset_index(drop=True)
    got = cudf.read_orc(buff, skiprows=skiprows)

    assert_eq(expected, got)


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
def test_chunked_orc_writer(
    datadir, tmpdir, reference_file, columns, compression
):
    pdf_fname = datadir / reference_file
    gdf_fname = tmpdir.join("chunked_gdf.orc")

    try:
        orcfile = pa.orc.ORCFile(pdf_fname)
    except Exception as excpr:
        if type(excpr).__name__ == "ArrowIOError":
            pytest.skip(".orc file is not found")
        else:
            print(type(excpr).__name__)

    pdf = orcfile.read(columns=columns).to_pandas()
    gdf = cudf.from_pandas(pdf)
    expect = pd.concat([pdf, pdf]).reset_index(drop=True)

    writer = ORCWriter(gdf_fname, compression=compression)
    writer.write_table(gdf)
    writer.write_table(gdf)
    writer.close()

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


@pytest.mark.parametrize(
    "dtypes",
    [
        {"c": str, "a": int},
        {"c": int, "a": str},
        {"c": int, "a": str, "b": float},
        {"c": str, "a": object},
    ],
)
def test_chunked_orc_writer_strings(tmpdir, dtypes):
    gdf_fname = tmpdir.join("chunked_gdf_strings.orc")

    gdf = cudf.datasets.randomdata(nrows=10, dtypes=dtypes, seed=1)
    pdf = gdf.to_pandas()
    expect = pd.concat([pdf, pdf]).reset_index(drop=True)
    writer = ORCWriter(gdf_fname)
    writer.write_table(gdf)
    writer.write_table(gdf)
    writer.close()

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
        "TestOrcFile.decimal.orc",
        "TestOrcFile.decimal.same.values.orc",
        "TestOrcFile.decimal.multiple.values.orc",
        # For addional information take look at PR 7034
        "TestOrcFile.decimal.runpos.issue.orc",
    ],
)
def test_orc_reader_decimal_type(datadir, orc_file):
    file_path = datadir / orc_file

    try:
        orcfile = pa.orc.ORCFile(file_path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    df = cudf.read_orc(file_path).to_pandas()

    assert_eq(pdf, df)


def test_orc_decimal_precision_fail(datadir):
    file_path = datadir / "TestOrcFile.int_decimal.precision_19.orc"

    try:
        orcfile = pa.orc.ORCFile(file_path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    # Max precision supported is 18 (Decimal64Dtype limit)
    # and the data has the precision 19. This test should be removed
    # once Decimal128Dtype is introduced.
    with pytest.raises(RuntimeError):
        cudf.read_orc(file_path)

    # Shouldn't cause failure if decimal column is not chosen to be read.
    pdf = orcfile.read(columns=["int"]).to_pandas()
    gdf = cudf.read_orc(file_path, columns=["int"])

    assert_eq(pdf, gdf)


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
        return np.isclose(value1, value2)

    return value1 == value2


@pytest.mark.parametrize("nrows", [1, 100, 6000000])
def test_orc_write_statistics(tmpdir, datadir, nrows):
    supported_stat_types = supported_numpy_dtypes + ["str"]
    # Can't write random bool columns until issue #6763 is fixed
    if nrows == 6000000:
        supported_stat_types.remove("bool")

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
    (file_stats, stripes_stats,) = cudf.io.orc.read_orc_statistics([fname])

    # check file stats
    for col in gdf:
        if "minimum" in file_stats[0][col]:
            stats_min = file_stats[0][col]["minimum"]
            actual_min = gdf[col].min()
            assert normalized_equals(actual_min, stats_min)
        if "maximum" in file_stats[0][col]:
            stats_max = file_stats[0][col]["maximum"]
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
    (file_stats, stripes_stats,) = cudf.io.orc.read_orc_statistics([fname])

    # check file stats
    col = "col_bool"
    if "true_count" in file_stats[0][col]:
        stats_true_count = file_stats[0][col]["true_count"]
        actual_true_count = gdf[col].sum()
        assert normalized_equals(actual_true_count, stats_true_count)

    if "number_of_values" in file_stats[0][col]:
        stats_valid_count = file_stats[0][col]["number_of_values"]
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


def test_orc_bool_encode_fail():
    np.random.seed(0)
    buffer = BytesIO()

    # Generate a boolean column longer than a single stripe
    fail_df = cudf.DataFrame({"col": gen_rand_series("bool", 600000)})
    # Invalidate the first row in the second stripe to break encoding
    fail_df["col"][500000] = None

    # Should throw instead of generating a file that is incompatible
    # with other readers (see issue #6763)
    with pytest.raises(RuntimeError):
        fail_df.to_orc(buffer)

    # Generate a boolean column that fits into a single stripe
    okay_df = cudf.DataFrame({"col": gen_rand_series("bool", 500000)})
    okay_df["col"][500000 - 1] = None
    # Invalid row is in the last row group of the stripe;
    # encoding is assumed to be correct
    okay_df.to_orc(buffer)

    # Also validate data
    pdf = pa.orc.ORCFile(buffer).read().to_pandas()
    assert_eq(okay_df, pdf)


def test_nanoseconds_overflow():
    buffer = BytesIO()
    # Use nanosecond values that take more than 32 bits to encode
    s = cudf.Series([710424008, -1338482640], dtype="datetime64[ns]")
    expected = cudf.DataFrame({"s": s})
    expected.to_orc(buffer)

    cudf_got = cudf.read_orc(buffer)
    assert_eq(expected, cudf_got)

    pyarrow_got = pa.orc.ORCFile(buffer).read()
    assert_eq(expected.to_pandas(), pyarrow_got.to_pandas())


def test_empty_dataframe():
    buffer = BytesIO()
    expected = cudf.DataFrame()
    expected.to_orc(buffer)

    # Raise error if column name is mentioned, but it doesn't exist.
    with pytest.raises(RuntimeError):
        cudf.read_orc(buffer, columns=["a"])

    got_df = cudf.read_orc(buffer)
    expected_pdf = pd.read_orc(buffer)

    assert_eq(expected, got_df)
    assert_eq(expected_pdf, got_df)


@pytest.mark.parametrize(
    "data", [[None, ""], ["", None], [None, None], ["", ""]]
)
def test_empty_string_columns(data):
    buffer = BytesIO()

    expected = cudf.DataFrame({"string": data}, dtype="str")
    expected.to_orc(buffer)

    expected_pdf = pd.read_orc(buffer)
    got_df = cudf.read_orc(buffer)

    assert_eq(expected, got_df)
    assert_eq(expected_pdf, got_df)


@pytest.mark.parametrize("scale", [-3, 0, 3])
def test_orc_writer_decimal(tmpdir, scale):
    np.random.seed(0)
    fname = tmpdir / "decimal.orc"

    expected = cudf.DataFrame({"dec_val": gen_rand_series("i", 100)})
    expected["dec_val"] = expected["dec_val"].astype(Decimal64Dtype(7, scale))

    expected.to_orc(fname)

    got = pd.read_orc(fname)
    assert_eq(expected.to_pandas()["dec_val"], got["dec_val"])


@pytest.mark.parametrize("num_rows", [1, 100, 3000])
def test_orc_reader_multiple_files(datadir, num_rows):

    path = datadir / "TestOrcFile.testSnappy.orc"

    df_1 = pd.read_orc(path)
    df_2 = pd.read_orc(path)
    df = pd.concat([df_1, df_2], ignore_index=True)

    gdf = cudf.read_orc(
        [path, path], engine="cudf", num_rows=num_rows
    ).to_pandas()

    # Slice rows out of the whole dataframe for comparison as PyArrow doesn't
    # have an API to read a subsection of rows from the file
    df = df[:num_rows]
    df = df.reset_index(drop=True)

    assert_eq(df, gdf)


def test_orc_reader_multi_file_single_stripe(datadir):

    path = datadir / "TestOrcFile.testSnappy.orc"

    # should raise an exception
    with pytest.raises(ValueError):
        cudf.read_orc([path, path], engine="cudf", stripes=[0])


def test_orc_reader_multi_file_multi_stripe(datadir):

    path = datadir / "TestOrcFile.testStripeLevelStats.orc"
    gdf = cudf.read_orc([path, path], engine="cudf", stripes=[[0, 1], [2]])
    pdf = pd.read_orc(path)
    assert_eq(pdf, gdf)


def test_orc_string_stream_offset_issue():
    size = 30000
    vals = {
        str(x): [decimal.Decimal(1)] * size if x != 0 else ["XYZ"] * size
        for x in range(0, 5)
    }
    df = cudf.DataFrame(vals)

    buffer = BytesIO()
    df.to_orc(buffer)

    assert_eq(df, cudf.read_orc(buffer))


def generate_list_struct_buff(size=28000):
    rd = random.Random(0)
    np.random.seed(seed=0)

    buff = BytesIO()

    schema = {
        "lvl3_list": po.Array(po.Array(po.Array(po.BigInt()))),
        "lvl1_list": po.Array(po.BigInt()),
        "lvl1_struct": po.Struct(**{"a": po.BigInt(), "b": po.BigInt()}),
        "lvl2_struct": po.Struct(
            **{
                "a": po.BigInt(),
                "lvl1_struct": po.Struct(
                    **{"c": po.BigInt(), "d": po.BigInt()}
                ),
            }
        ),
        "list_nests_struct": po.Array(
            po.Array(po.Struct(**{"a": po.BigInt(), "b": po.BigInt()}))
        ),
        "struct_nests_list": po.Struct(
            **{
                "struct": po.Struct(**{"a": po.BigInt(), "b": po.BigInt()}),
                "list": po.Array(po.BigInt()),
            }
        ),
    }

    schema = po.Struct(**schema)

    lvl3_list = [
        [
            [
                [
                    rd.choice([None, np.random.randint(1, 3)])
                    for z in range(np.random.randint(1, 3))
                ]
                for z in range(np.random.randint(0, 3))
            ]
            for y in range(np.random.randint(0, 3))
        ]
        for x in range(size)
    ]
    lvl1_list = [
        [
            rd.choice([None, np.random.randint(0, 3)])
            for y in range(np.random.randint(1, 4))
        ]
        for x in range(size)
    ]
    lvl1_struct = [
        (np.random.randint(0, 3), np.random.randint(0, 3)) for x in range(size)
    ]
    lvl2_struct = [
        (
            rd.choice([None, np.random.randint(0, 3)]),
            (
                rd.choice([None, np.random.randint(0, 3)]),
                np.random.randint(0, 3),
            ),
        )
        for x in range(size)
    ]
    list_nests_struct = [
        [
            [rd.choice(lvl1_struct), rd.choice(lvl1_struct)]
            for y in range(np.random.randint(1, 4))
        ]
        for x in range(size)
    ]
    struct_nests_list = [(lvl1_struct[x], lvl1_list[x]) for x in range(size)]

    df = pd.DataFrame(
        {
            "lvl3_list": lvl3_list,
            "lvl1_list": lvl1_list,
            "lvl1_struct": lvl1_struct,
            "lvl2_struct": lvl2_struct,
            "list_nests_struct": list_nests_struct,
            "struct_nests_list": struct_nests_list,
        }
    )

    writer = po.Writer(buff, schema, stripe_size=1024)
    tuples = list(
        map(
            lambda x: (None,) if x[0] is pd.NA else x,
            list(df.itertuples(index=False, name=None)),
        )
    )
    writer.writerows(tuples)
    writer.close()

    return buff


list_struct_buff = generate_list_struct_buff()


@pytest.mark.parametrize(
    "columns",
    [
        None,
        ["lvl3_list", "list_nests_struct", "lvl2_struct", "struct_nests_list"],
        ["lvl2_struct", "lvl1_struct"],
    ],
)
@pytest.mark.parametrize("num_rows", [0, 15, 1005, 10561, 28000])
@pytest.mark.parametrize("use_index", [True, False])
@pytest.mark.parametrize("skip_rows", [0, 101, 1007])
def test_lists_struct_nests(
    columns, num_rows, use_index, skip_rows,
):

    has_lists = (
        any("list" in col_name for col_name in columns) if columns else True
    )

    if has_lists and skip_rows > 0:
        with pytest.raises(
            RuntimeError, match="skip_rows is not supported by list column"
        ):
            cudf.read_orc(
                list_struct_buff,
                columns=columns,
                num_rows=num_rows,
                use_index=use_index,
                skiprows=skip_rows,
            )
    else:
        gdf = cudf.read_orc(
            list_struct_buff,
            columns=columns,
            num_rows=num_rows,
            use_index=use_index,
            skiprows=skip_rows,
        )

        pyarrow_tbl = pyarrow.orc.ORCFile(list_struct_buff).read()

        pyarrow_tbl = (
            pyarrow_tbl[skip_rows : skip_rows + num_rows]
            if columns is None
            else pyarrow_tbl.select(columns)[skip_rows : skip_rows + num_rows]
        )

        if num_rows > 0:
            assert_eq(True, pyarrow_tbl.equals(gdf.to_arrow()))
        else:
            assert_eq(pyarrow_tbl.to_pandas(), gdf)


@pytest.mark.parametrize(
    "data", [["_col0"], ["FakeName", "_col0", "TerriblyFakeColumnName"]]
)
def test_orc_reader_decimal(datadir, data):
    path = datadir / "TestOrcFile.decimal.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(
        path, engine="cudf", decimal_cols_as_float=data
    ).to_pandas()

    # Convert the decimal dtype from PyArrow to float64 for comparison to cuDF
    # This is because cuDF returns as float64
    pdf = pdf.apply(pd.to_numeric)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("data", [["InvalidColumnName"]])
def test_orc_reader_decimal_invalid_column(datadir, data):
    path = datadir / "TestOrcFile.decimal.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(
        path, engine="cudf", decimal_cols_as_float=data
    ).to_pandas()

    # Since the `decimal_cols_as_float` column name
    # is invalid, this should be a decimal
    assert_eq(pdf, gdf)
