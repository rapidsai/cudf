# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import datetime
import decimal
import os
import random
from io import BytesIO
from string import ascii_lowercase

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pyorc
import pytest

import cudf
from cudf.io.orc import ORCWriter
from cudf.testing import assert_frame_equal
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

    assert_frame_equal(cudf.from_pandas(expect), got, check_categorical=False)


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
    got = cudf.read_orc(path).fillna(0)

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
    gdf = cudf.read_orc(path, use_index=use_index)

    assert_eq(pdf, gdf, check_categorical=False)


def test_orc_reader_strings(datadir):
    path = datadir / "TestOrcFile.testStringAndBinaryStatistics.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    expect = orcfile.read(columns=["string1"])
    got = cudf.read_orc(path, columns=["string1"])

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


@pytest.mark.filterwarnings("ignore:Using CPU")
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


@pytest.mark.filterwarnings("ignore:Using CPU")
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
    assert_eq(pdf, gdf, check_categorical=False, check_index_type=True)

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
        gdf,
        cudf.concat([pdf.head(15000), pdf.tail(10000)], ignore_index=True),
        check_index_type=True,
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
    gdf = cudf.read_orc(path, skiprows=skiprows, num_rows=num_rows)

    # Slice rows out of the whole dataframe for comparison as PyArrow doesn't
    # have an API to read a subsection of rows from the file
    pdf = pdf[skiprows : skiprows + num_rows]
    pdf = pdf.reset_index(drop=True)

    assert_eq(pdf, gdf)


def test_orc_read_skiprows():
    buff = BytesIO()
    data = [
        True,
        False,
        True,
        False,
        None,
        True,
        True,
        True,
        False,
        None,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    writer = pyorc.Writer(buff, pyorc.Struct(a=pyorc.Boolean()))
    writer.writerows([(d,) for d in data])
    writer.close()

    # testing 10 skiprows due to a boolean specific bug fix that didn't
    # repro for other sizes of data
    skiprows = 10

    expected = cudf.read_orc(buff)[skiprows:].reset_index(drop=True)
    got = cudf.read_orc(buff, skiprows=skiprows)
    assert_eq(expected, got)


def test_orc_reader_uncompressed_block(datadir):
    path = datadir / "uncompressed_snappy.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    expect = orcfile.read().to_pandas()
    got = cudf.read_orc(path)

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
    got = cudf.read_orc(path, num_rows=1)

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

    expect = cudf.from_pandas(orcfile.read(columns=columns).to_pandas())
    expect.to_orc(gdf_fname.strpath, compression=compression)
    got = cudf.from_pandas(
        pa.orc.ORCFile(gdf_fname).read(columns=columns).to_pandas()
    )

    assert_frame_equal(expect, got)


@pytest.mark.parametrize("stats_freq", ["NONE", "STRIPE", "ROWGROUP"])
def test_orc_writer_statistics_frequency(datadir, tmpdir, stats_freq):
    reference_file = "TestOrcFile.demo-12-zlib.orc"
    pdf_fname = datadir / reference_file
    gdf_fname = tmpdir.join("gdf.orc")

    try:
        orcfile = pa.orc.ORCFile(pdf_fname)
    except Exception as excpr:
        if type(excpr).__name__ == "ArrowIOError":
            pytest.skip(".orc file is not found")
        else:
            print(type(excpr).__name__)

    expect = cudf.from_pandas(orcfile.read().to_pandas())
    expect.to_orc(gdf_fname.strpath, statistics=stats_freq)
    got = cudf.from_pandas(pa.orc.ORCFile(gdf_fname).read().to_pandas())

    assert_frame_equal(expect, got)


@pytest.mark.parametrize("stats_freq", ["NONE", "STRIPE", "ROWGROUP"])
def test_chunked_orc_writer_statistics_frequency(datadir, tmpdir, stats_freq):
    reference_file = "TestOrcFile.test1.orc"
    pdf_fname = datadir / reference_file
    gdf_fname = tmpdir.join("chunked_gdf.orc")

    try:
        orcfile = pa.orc.ORCFile(pdf_fname)
    except Exception as excpr:
        if type(excpr).__name__ == "ArrowIOError":
            pytest.skip(".orc file is not found")
        else:
            print(type(excpr).__name__)

    columns = [
        "boolean1",
        "byte1",
        "short1",
        "int1",
        "long1",
        "float1",
        "double1",
    ]
    pdf = orcfile.read(columns=columns).to_pandas()
    gdf = cudf.from_pandas(pdf)
    expect = pd.concat([pdf, pdf]).reset_index(drop=True)

    writer = ORCWriter(gdf_fname, statistics=stats_freq)
    writer.write_table(gdf)
    writer.write_table(gdf)
    writer.close()

    got = pa.orc.ORCFile(gdf_fname).read().to_pandas()

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
    assert_frame_equal(cudf.from_pandas(expect), cudf.from_pandas(got))


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
    assert_eq(cudf.read_orc(cudf_path), df_select)


@pytest.mark.parametrize(
    "orc_file",
    [
        "TestOrcFile.decimal.orc",
        "TestOrcFile.decimal.same.values.orc",
        "TestOrcFile.decimal.multiple.values.orc",
        # For additional information take look at PR 7034
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

    # Shouldn't cause failure if decimal column is not chosen to be read.
    pdf = orcfile.read(columns=["int"]).to_pandas()
    gdf = cudf.read_orc(file_path, columns=["int"])

    assert_eq(pdf, gdf)


# For additional information take look at PR 6636 and 6702
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


def test_orc_reader_tzif_timestamps(datadir):
    # Contains timstamps in the range covered by the TZif file
    # Other timedate tests only cover "future" times
    path = datadir / "TestOrcFile.lima_timezone.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(path).to_pandas()

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
    # need naive time object for numpy to convert to datetime64
    if isinstance(value1, datetime.datetime):
        value1 = value1.replace(tzinfo=None)
    if isinstance(value2, datetime.datetime):
        value2 = value2.replace(tzinfo=None)

    if isinstance(value1, (datetime.datetime, np.datetime64)):
        value1 = np.datetime64(value1, "ms")
    if isinstance(value2, (datetime.datetime, np.datetime64)):
        value2 = np.datetime64(value2, "ms")

    # Compare integers with floats now
    if isinstance(value1, float) or isinstance(value2, float):
        return np.isclose(value1, value2)

    return value1 == value2


@pytest.mark.parametrize("stats_freq", ["STRIPE", "ROWGROUP"])
@pytest.mark.parametrize("nrows", [1, 100, 6000000])
def test_orc_write_statistics(tmpdir, datadir, nrows, stats_freq):
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
    gdf.to_orc(fname.strpath, statistics=stats_freq)

    # Read back written ORC's statistics
    orc_file = pa.orc.ORCFile(fname)
    (
        file_stats,
        stripes_stats,
    ) = cudf.io.orc.read_orc_statistics([fname])

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
        if "number_of_values" in file_stats[0][col]:
            stats_num_vals = file_stats[0][col]["number_of_values"]
            actual_num_vals = gdf[col].count()
            assert stats_num_vals == actual_num_vals

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

            if "number_of_values" in stripes_stats[stripe_idx][col]:
                stats_num_vals = stripes_stats[stripe_idx][col][
                    "number_of_values"
                ]
                actual_num_vals = stripe_df[col].count()
                assert stats_num_vals == actual_num_vals


@pytest.mark.parametrize("stats_freq", ["STRIPE", "ROWGROUP"])
@pytest.mark.parametrize("nrows", [2, 100, 6000000])
def test_orc_chunked_write_statistics(tmpdir, datadir, nrows, stats_freq):
    supported_stat_types = supported_numpy_dtypes + ["str"]
    # Can't write random bool columns until issue #6763 is fixed
    if nrows == 6000000:
        supported_stat_types.remove("bool")

    gdf_fname = tmpdir.join("chunked_stats.orc")
    writer = ORCWriter(gdf_fname)

    max_char_length = 1000 if nrows < 10000 else 100

    # Make a dataframe
    gdf = cudf.DataFrame(
        {
            "col_"
            + str(dtype): gen_rand_series(
                dtype,
                int(nrows / 2),
                has_nulls=True,
                low=0,
                high=max_char_length,
            )
            for dtype in supported_stat_types
        }
    )

    pdf1 = gdf.to_pandas()
    writer.write_table(gdf)
    # gdf is specifically being reused here to ensure the data is destroyed
    # before the next write_table call to ensure the data is persisted inside
    # write and no pointers are saved into the original table
    gdf = cudf.DataFrame(
        {
            "col_"
            + str(dtype): gen_rand_series(
                dtype,
                int(nrows / 2),
                has_nulls=True,
                low=0,
                high=max_char_length,
            )
            for dtype in supported_stat_types
        }
    )
    pdf2 = gdf.to_pandas()
    writer.write_table(gdf)
    writer.close()

    # pandas is unable to handle min/max of string col with nulls
    expect = cudf.DataFrame(pd.concat([pdf1, pdf2]).reset_index(drop=True))

    # Read back written ORC's statistics
    orc_file = pa.orc.ORCFile(gdf_fname)
    (
        file_stats,
        stripes_stats,
    ) = cudf.io.orc.read_orc_statistics([gdf_fname])

    # check file stats
    for col in expect:
        if "minimum" in file_stats[0][col]:
            stats_min = file_stats[0][col]["minimum"]
            actual_min = expect[col].min()
            assert normalized_equals(actual_min, stats_min)
        if "maximum" in file_stats[0][col]:
            stats_max = file_stats[0][col]["maximum"]
            actual_max = expect[col].max()
            assert normalized_equals(actual_max, stats_max)
        if "number_of_values" in file_stats[0][col]:
            stats_num_vals = file_stats[0][col]["number_of_values"]
            actual_num_vals = expect[col].count()
            assert stats_num_vals == actual_num_vals

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

            if "number_of_values" in stripes_stats[stripe_idx][col]:
                stats_num_vals = stripes_stats[stripe_idx][col][
                    "number_of_values"
                ]
                actual_num_vals = stripe_df[col].count()
                assert stats_num_vals == actual_num_vals


@pytest.mark.parametrize("nrows", [1, 100, 6000000])
def test_orc_write_bool_statistics(tmpdir, datadir, nrows):
    # Make a dataframe
    gdf = cudf.DataFrame({"col_bool": gen_rand_series("bool", nrows)})
    fname = tmpdir.join("gdf.orc")

    # Write said dataframe to ORC with cuDF
    gdf.to_orc(fname.strpath)

    # Read back written ORC's statistics
    orc_file = pa.orc.ORCFile(fname)
    (
        file_stats,
        stripes_stats,
    ) = cudf.io.orc.read_orc_statistics([fname])

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
    gdf = cudf.read_orc(path).to_pandas()
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

    assert_eq(okay_df.to_pandas(nullable=True), pdf)


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
    assert_eq(
        expected_pdf,
        got_df.to_pandas(nullable=True)
        if expected_pdf["string"].dtype == pd.StringDtype()
        else got_df,
    )


@pytest.mark.parametrize("scale", [-3, 0, 3])
@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
)
def test_orc_writer_decimal(tmpdir, scale, decimal_type):
    np.random.seed(0)
    fname = tmpdir / "decimal.orc"

    expected = cudf.DataFrame({"dec_val": gen_rand_series("i", 100)})
    expected["dec_val"] = expected["dec_val"].astype(decimal_type(7, scale))

    expected.to_orc(fname)

    got = pd.read_orc(fname)
    assert_eq(expected.to_pandas()["dec_val"], got["dec_val"])


@pytest.mark.parametrize("num_rows", [1, 100, 3000])
def test_orc_reader_multiple_files(datadir, num_rows):

    path = datadir / "TestOrcFile.testSnappy.orc"

    df_1 = pd.read_orc(path)
    df_2 = pd.read_orc(path)
    df = pd.concat([df_1, df_2], ignore_index=True)

    gdf = cudf.read_orc([path, path], num_rows=num_rows).to_pandas()

    # Slice rows out of the whole dataframe for comparison as PyArrow doesn't
    # have an API to read a subsection of rows from the file
    df = df[:num_rows]
    df = df.reset_index(drop=True)

    assert_eq(df, gdf)


def test_orc_reader_multi_file_single_stripe(datadir):

    path = datadir / "TestOrcFile.testSnappy.orc"

    # should raise an exception
    with pytest.raises(ValueError):
        cudf.read_orc([path, path], stripes=[0])


def test_orc_reader_multi_file_multi_stripe(datadir):

    path = datadir / "TestOrcFile.testStripeLevelStats.orc"
    gdf = cudf.read_orc([path, path], stripes=[[0, 1], [2]])
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


# Data is generated using pyorc module
def generate_list_struct_buff(size=100_000):
    rd = random.Random(1)
    np.random.seed(seed=1)

    buff = BytesIO()

    schema = {
        "lvl3_list": pyorc.Array(pyorc.Array(pyorc.Array(pyorc.BigInt()))),
        "lvl1_list": pyorc.Array(pyorc.BigInt()),
        "lvl1_struct": pyorc.Struct(
            **{"a": pyorc.BigInt(), "b": pyorc.BigInt()}
        ),
        "lvl2_struct": pyorc.Struct(
            **{
                "a": pyorc.BigInt(),
                "lvl1_struct": pyorc.Struct(
                    **{"c": pyorc.BigInt(), "d": pyorc.BigInt()}
                ),
            }
        ),
        "list_nests_struct": pyorc.Array(
            pyorc.Array(
                pyorc.Struct(**{"a": pyorc.BigInt(), "b": pyorc.BigInt()})
            )
        ),
        "struct_nests_list": pyorc.Struct(
            **{
                "struct": pyorc.Struct(
                    **{"a": pyorc.BigInt(), "b": pyorc.BigInt()}
                ),
                "list": pyorc.Array(pyorc.BigInt()),
            }
        ),
    }

    schema = pyorc.Struct(**schema)

    lvl3_list = [
        rd.choice(
            [
                None,
                [
                    [
                        [
                            rd.choice([None, np.random.randint(1, 3)])
                            for z in range(np.random.randint(1, 3))
                        ]
                        for z in range(np.random.randint(0, 3))
                    ]
                    for y in range(np.random.randint(0, 3))
                ],
            ]
        )
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
        rd.choice([None, (np.random.randint(0, 3), np.random.randint(0, 3))])
        for x in range(size)
    ]
    lvl2_struct = [
        rd.choice(
            [
                None,
                (
                    rd.choice([None, np.random.randint(0, 3)]),
                    (
                        rd.choice([None, np.random.randint(0, 3)]),
                        np.random.randint(0, 3),
                    ),
                ),
            ]
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

    writer = pyorc.Writer(buff, schema, stripe_size=1024)
    tuples = list(
        map(
            lambda x: (None,) if x[0] is pd.NA else x,
            list(df.itertuples(index=False, name=None)),
        )
    )
    writer.writerows(tuples)
    writer.close()

    return buff


@pytest.fixture(scope="module")
def list_struct_buff():
    return generate_list_struct_buff()


@pytest.mark.parametrize(
    "columns",
    [
        None,
        ["lvl3_list", "list_nests_struct", "lvl2_struct", "struct_nests_list"],
        ["lvl2_struct", "lvl1_struct"],
    ],
)
@pytest.mark.parametrize("num_rows", [0, 15, 1005, 10561, 100_000])
@pytest.mark.parametrize("use_index", [True, False])
def test_lists_struct_nests(columns, num_rows, use_index, list_struct_buff):

    gdf = cudf.read_orc(
        list_struct_buff,
        columns=columns,
        num_rows=num_rows,
        use_index=use_index,
    )

    pyarrow_tbl = pyarrow.orc.ORCFile(list_struct_buff).read()

    pyarrow_tbl = (
        pyarrow_tbl[:num_rows]
        if columns is None
        else pyarrow_tbl.select(columns)[:num_rows]
    )

    if num_rows > 0:
        assert pyarrow_tbl.equals(gdf.to_arrow())
    else:
        assert_eq(pyarrow_tbl.to_pandas(), gdf)


@pytest.mark.parametrize("columns", [None, ["lvl1_struct"], ["lvl1_list"]])
def test_skip_rows_for_nested_types(columns, list_struct_buff):
    with pytest.raises(
        RuntimeError, match="skip_rows is not supported by nested column"
    ):
        cudf.read_orc(
            list_struct_buff,
            columns=columns,
            use_index=True,
            skiprows=5,
        )


def test_pyspark_struct(datadir):
    path = datadir / "TestOrcFile.testPySparkStruct.orc"

    pdf = pa.orc.ORCFile(path).read().to_pandas()
    gdf = cudf.read_orc(path)

    assert_eq(pdf, gdf)


def gen_map_buff(size=10000):
    from string import ascii_letters as al

    rd = random.Random(1)
    np.random.seed(seed=1)

    buff = BytesIO()

    schema = {
        "lvl1_map": pyorc.Map(key=pyorc.String(), value=pyorc.BigInt()),
        "lvl2_map": pyorc.Map(
            key=pyorc.String(), value=pyorc.Array(pyorc.BigInt())
        ),
        "lvl2_struct_map": pyorc.Map(
            key=pyorc.String(),
            value=pyorc.Struct(**{"a": pyorc.BigInt(), "b": pyorc.BigInt()}),
        ),
    }

    schema = pyorc.Struct(**schema)

    lvl1_map = [
        rd.choice(
            [
                None,
                [
                    (
                        rd.choice(al),
                        rd.choice([None, np.random.randint(1, 1500)]),
                    )
                    for y in range(2)
                ],
            ]
        )
        for x in range(size)
    ]
    lvl2_map = [
        rd.choice(
            [
                None,
                [
                    (
                        rd.choice(al),
                        rd.choice(
                            [
                                None,
                                [
                                    rd.choice(
                                        [None, np.random.randint(1, 1500)]
                                    )
                                    for z in range(5)
                                ],
                            ]
                        ),
                    )
                    for y in range(2)
                ],
            ]
        )
        for x in range(size)
    ]
    lvl2_struct_map = [
        rd.choice(
            [
                None,
                [
                    (
                        rd.choice(al),
                        rd.choice(
                            [
                                None,
                                (
                                    rd.choice(
                                        [None, np.random.randint(1, 1500)]
                                    ),
                                    rd.choice(
                                        [None, np.random.randint(1, 1500)]
                                    ),
                                ),
                            ]
                        ),
                    )
                    for y in range(2)
                ],
            ]
        )
        for x in range(size)
    ]

    pdf = pd.DataFrame(
        {
            "lvl1_map": lvl1_map,
            "lvl2_map": lvl2_map,
            "lvl2_struct_map": lvl2_struct_map,
        }
    )
    writer = pyorc.Writer(
        buff, schema, stripe_size=1024, compression=pyorc.CompressionKind.NONE
    )
    tuples = list(
        map(
            lambda x: (None,) if x[0] is pd.NA else x,
            list(pdf.itertuples(index=False, name=None)),
        )
    )

    writer.writerows(tuples)
    writer.close()

    return buff


map_buff = gen_map_buff(size=100000)


@pytest.mark.parametrize(
    "columns",
    [None, ["lvl1_map", "lvl2_struct_map"], ["lvl2_struct_map", "lvl2_map"]],
)
@pytest.mark.parametrize("num_rows", [0, 15, 1005, 10561, 100000])
@pytest.mark.parametrize("use_index", [True, False])
def test_map_type_read(columns, num_rows, use_index):
    tbl = pa.orc.ORCFile(map_buff).read()

    lvl1_map = (
        tbl["lvl1_map"]
        .combine_chunks()
        .view(pa.list_(pa.struct({"key": pa.string(), "value": pa.int64()})))
    )
    lvl2_map = (
        tbl["lvl2_map"]
        .combine_chunks()
        .view(
            pa.list_(
                pa.struct({"key": pa.string(), "value": pa.list_(pa.int64())})
            )
        )
    )
    lvl2_struct_map = (
        tbl["lvl2_struct_map"]
        .combine_chunks()
        .view(
            pa.list_(
                pa.struct(
                    {
                        "key": pa.string(),
                        "value": pa.struct({"a": pa.int64(), "b": pa.int64()}),
                    }
                )
            )
        )
    )

    expected_tbl = pa.table(
        {
            "lvl1_map": lvl1_map,
            "lvl2_map": lvl2_map,
            "lvl2_struct_map": lvl2_struct_map,
        }
    )
    gdf = cudf.read_orc(
        map_buff, columns=columns, num_rows=num_rows, use_index=use_index
    )

    expected_tbl = (
        expected_tbl[:num_rows]
        if columns is None
        else expected_tbl.select(columns)[:num_rows]
    )

    if num_rows > 0:
        assert expected_tbl.equals(gdf.to_arrow())
    else:
        assert_eq(expected_tbl.to_pandas(), gdf)


def test_orc_reader_decimal(datadir):
    path = datadir / "TestOrcFile.decimal.orc"
    try:
        orcfile = pa.orc.ORCFile(path)
    except pa.ArrowIOError as e:
        pytest.skip(".orc file is not found: %s" % e)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(path).to_pandas()

    assert_eq(pdf, gdf)


# This test case validates the issue raised in #8665,
# please check the issue for more details.
def test_orc_timestamp_read(datadir):
    path = datadir / "TestOrcFile.timestamp.issue.orc"

    pdf = pd.read_orc(path)
    gdf = cudf.read_orc(path)

    assert_eq(pdf, gdf)


def dec(num):
    return decimal.Decimal(str(num))


@pytest.mark.parametrize(
    "data",
    [
        # basic + nested strings
        {
            "lls": [[["a"], ["bb"]] * 5 for i in range(12345)],
            "lls2": [[["ccc", "dddd"]] * 6 for i in range(12345)],
            "ls_dict": [["X"] * 7 for i in range(12345)],
            "ls_direct": [[str(i)] * 9 for i in range(12345)],
            "li": [[i] * 11 for i in range(12345)],
            "lf": [[i * 0.5] * 13 for i in range(12345)],
            "ld": [[dec(i / 2)] * 15 for i in range(12345)],
        },
        # with nulls
        {
            "ls": [
                [str(i) if i % 5 else None, str(2 * i)] if i % 2 else None
                for i in range(12345)
            ],
            "li": [[i, i * i, i % 2] if i % 3 else None for i in range(12345)],
            "ld": [
                [dec(i), dec(i / 2) if i % 7 else None] if i % 5 else None
                for i in range(12345)
            ],
        },
        # with empty elements
        {
            "ls": [
                [str(i), str(2 * i)] if i % 2 else [] for i in range(12345)
            ],
            "lls": [
                [[str(i), str(2 * i)]] if i % 2 else [[], []]
                for i in range(12345)
            ],
            "li": [[i, i * i, i % 2] if i % 3 else [] for i in range(12345)],
            "lli": [
                [[i], [i * i], [i % 2]] if i % 3 else [[]]
                for i in range(12345)
            ],
            "ld": [
                [dec(i), dec(i / 2)] if i % 5 else [] for i in range(12345)
            ],
        },
        # variable list lengths
        {
            "ls": [[str(i)] * i for i in range(123)],
            "li": [[i, i * i] * i for i in range(123)],
            "ld": [[dec(i), dec(i / 2)] * i for i in range(123)],
        },
        # many child elements (more that max_stripe_rows)
        {"li": [[i] * 1100 for i in range(11000)]},
    ],
)
def test_orc_writer_lists(data):
    pdf_in = pd.DataFrame(data)

    buffer = BytesIO()
    cudf.from_pandas(pdf_in).to_orc(
        buffer, stripe_size_rows=2048, row_index_stride=512
    )

    pdf_out = pa.orc.ORCFile(buffer).read().to_pandas()
    assert_eq(pdf_out, pdf_in)


def test_chunked_orc_writer_lists():
    num_rows = 12345
    pdf_in = pd.DataFrame(
        {
            "ls": [[str(i), str(2 * i)] for i in range(num_rows)],
            "ld": [[dec(i / 2)] * 5 for i in range(num_rows)],
        }
    )

    gdf = cudf.from_pandas(pdf_in)
    expect = pd.concat([pdf_in, pdf_in]).reset_index(drop=True)

    buffer = BytesIO()
    writer = ORCWriter(buffer)
    writer.write_table(gdf)
    writer.write_table(gdf)
    writer.close()

    got = pa.orc.ORCFile(buffer).read().to_pandas()
    assert_eq(expect, got)


def test_writer_timestamp_stream_size(datadir, tmpdir):
    pdf_fname = datadir / "TestOrcFile.largeTimestamps.orc"
    gdf_fname = tmpdir.join("gdf.orc")

    try:
        orcfile = pa.orc.ORCFile(pdf_fname)
    except Exception as excpr:
        if type(excpr).__name__ == "ArrowIOError":
            pytest.skip(".orc file is not found")
        else:
            print(type(excpr).__name__)

    expect = orcfile.read().to_pandas()
    cudf.from_pandas(expect).to_orc(gdf_fname.strpath)
    got = pa.orc.ORCFile(gdf_fname).read().to_pandas()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "fname",
    [
        "TestOrcFile.NoIndStrm.StructWithNoNulls.orc",
        "TestOrcFile.NoIndStrm.StructAndIntWithNulls.orc",
        "TestOrcFile.NoIndStrm.StructAndIntWithNulls.TwoStripes.orc",
        "TestOrcFile.NoIndStrm.IntWithNulls.orc",
    ],
)
def test_no_row_group_index_orc_read(datadir, fname):
    fpath = datadir / fname

    expect = pa.orc.ORCFile(fpath).read()
    got = cudf.read_orc(fpath)

    assert expect.equals(got.to_arrow())


def test_names_in_struct_dtype_nesting(datadir):
    fname = datadir / "TestOrcFile.NestedStructDataFrame.orc"

    expect = pa.orc.ORCFile(fname).read()
    got = cudf.read_orc(fname)

    # test dataframes
    assert expect.equals(got.to_arrow())

    edf = cudf.DataFrame(expect.to_pandas())
    # test schema
    assert edf.dtypes.equals(got.dtypes)


@pytest.mark.filterwarnings("ignore:.*struct.*experimental")
def test_writer_lists_structs(list_struct_buff):
    df_in = cudf.read_orc(list_struct_buff)

    buff = BytesIO()
    df_in.to_orc(buff)

    pyarrow_tbl = pyarrow.orc.ORCFile(buff).read()

    assert pyarrow_tbl.equals(df_in.to_arrow())


@pytest.mark.filterwarnings("ignore:.*struct.*experimental")
@pytest.mark.parametrize(
    "data",
    [
        {
            "with_pd": [
                [i if i % 3 else None] if i < 9999 or i > 20001 else None
                for i in range(21000)
            ],
            "no_pd": [
                [i if i % 3 else None] if i < 9999 or i > 20001 else []
                for i in range(21000)
            ],
        },
    ],
)
def test_orc_writer_lists_empty_rg(data):
    pdf_in = pd.DataFrame(data)
    buffer = BytesIO()
    cudf_in = cudf.from_pandas(pdf_in)

    cudf_in.to_orc(buffer)

    df = cudf.read_orc(buffer)
    assert_eq(df, cudf_in)

    pdf_out = pa.orc.ORCFile(buffer).read().to_pandas()
    assert_eq(pdf_in, pdf_out)


def test_statistics_sum_overflow():
    maxint64 = np.iinfo(np.int64).max
    minint64 = np.iinfo(np.int64).min

    buff = BytesIO()
    with pyorc.Writer(
        buff,
        pyorc.Struct(a=pyorc.BigInt(), b=pyorc.BigInt(), c=pyorc.BigInt()),
    ) as writer:
        writer.write((maxint64, minint64, minint64))
        writer.write((1, -1, 1))

    file_stats, stripe_stats = cudf.io.orc.read_orc_statistics([buff])
    assert file_stats[0]["a"].get("sum") is None
    assert file_stats[0]["b"].get("sum") is None
    assert file_stats[0]["c"].get("sum") == minint64 + 1

    assert stripe_stats[0]["a"].get("sum") is None
    assert stripe_stats[0]["b"].get("sum") is None
    assert stripe_stats[0]["c"].get("sum") == minint64 + 1


def test_empty_statistics():
    buff = BytesIO()
    orc_schema = pyorc.Struct(
        a=pyorc.BigInt(),
        b=pyorc.Double(),
        c=pyorc.String(),
        d=pyorc.Decimal(11, 2),
        e=pyorc.Date(),
        f=pyorc.Timestamp(),
        g=pyorc.Boolean(),
        h=pyorc.Binary(),
        i=pyorc.BigInt(),
        # One column with non null value, else cudf/pyorc readers crash
    )
    data = tuple([None] * (len(orc_schema.fields) - 1) + [1])
    with pyorc.Writer(buff, orc_schema) as writer:
        writer.write(data)

    got = cudf.io.orc.read_orc_statistics([buff])

    # Check for both file and stripe stats
    for stats in got:
        # Similar expected stats for the first 6 columns in this case
        for col_name in ascii_lowercase[:6]:
            assert stats[0][col_name].get("number_of_values") == 0
            assert stats[0][col_name].get("has_null") is True
            assert stats[0][col_name].get("minimum") is None
            assert stats[0][col_name].get("maximum") is None
        for col_name in ascii_lowercase[:3]:
            assert stats[0][col_name].get("sum") == 0
        # Sum for decimal column is a string
        assert stats[0]["d"].get("sum") == "0"

        assert stats[0]["g"].get("number_of_values") == 0
        assert stats[0]["g"].get("has_null") is True
        assert stats[0]["g"].get("true_count") == 0
        assert stats[0]["g"].get("false_count") == 0

        assert stats[0]["h"].get("number_of_values") == 0
        assert stats[0]["h"].get("has_null") is True
        assert stats[0]["h"].get("sum") == 0

        assert stats[0]["i"].get("number_of_values") == 1
        assert stats[0]["i"].get("has_null") is False
        assert stats[0]["i"].get("minimum") == 1
        assert stats[0]["i"].get("maximum") == 1
        assert stats[0]["i"].get("sum") == 1


@pytest.mark.filterwarnings("ignore:.*struct.*experimental")
@pytest.mark.parametrize(
    "equivalent_columns",
    [
        (["lvl1_struct.a", "lvl1_struct.b"], ["lvl1_struct"]),
        (["lvl1_struct", "lvl1_struct.a"], ["lvl1_struct"]),
        (["lvl1_struct.a", "lvl1_struct"], ["lvl1_struct"]),
        (["lvl1_struct.b", "lvl1_struct.a"], ["lvl1_struct.b", "lvl1_struct"]),
        (["lvl2_struct.lvl1_struct", "lvl2_struct"], ["lvl2_struct"]),
        (
            ["lvl2_struct.a", "lvl2_struct.lvl1_struct.c", "lvl2_struct"],
            ["lvl2_struct"],
        ),
    ],
)
def test_select_nested(list_struct_buff, equivalent_columns):
    # The two column selections should be equivalent
    df_cols1 = cudf.read_orc(list_struct_buff, columns=equivalent_columns[0])
    df_cols2 = cudf.read_orc(list_struct_buff, columns=equivalent_columns[1])
    assert_eq(df_cols1, df_cols2)


def test_orc_writer_rle_stream_size(datadir, tmpdir):
    original = datadir / "TestOrcFile.int16.rle.size.orc"
    reencoded = tmpdir.join("int16_map.orc")

    df = cudf.read_orc(original)
    df.to_orc(reencoded)

    # Segfaults when RLE stream sizes don't account for varint length
    pa_out = pa.orc.ORCFile(reencoded).read()
    assert_eq(df.to_pandas(), pa_out)


def test_empty_columns():
    buffer = BytesIO()
    # string and decimal columns have additional steps that need to be skipped
    expected = cudf.DataFrame(
        {
            "string": cudf.Series([], dtype="str"),
            "decimal": cudf.Series([], dtype=cudf.Decimal64Dtype(10, 1)),
        }
    )
    expected.to_orc(buffer, compression="snappy")

    got_df = cudf.read_orc(buffer)
    assert_eq(expected, got_df)
