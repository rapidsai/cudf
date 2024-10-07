# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import itertools
import pickle

import msgpack
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import _utils as utils, assert_eq


@pytest.mark.parametrize(
    "df",
    [
        lambda: cudf.Index([1, 2, 3]),
        lambda: cudf.Index([1.0, 2.0, 3.0]),
        lambda: cudf.Series([1, 2, 3]),
        lambda: cudf.Series([1, 2, 3], index=[4, 5, 6]),
        lambda: cudf.Series([1, None, 3]),
        lambda: cudf.Series([1, 2, 3], index=[4, 5, None]),
        lambda: cudf.Series([1, 2, 3])[:2],
        lambda: cudf.Series([1, 2, 3])[:2]._column,
        lambda: cudf.Series(["a", "bb", "ccc"]),
        lambda: cudf.Series(["a", None, "ccc"]),
        lambda: cudf.Series(
            [
                {"a": ({"b": [1, 2, 3], "c": [4, 5, 6]}, {"d": [2, 4, 6]})},
                {"e": ({"b": [0, 2, 4], "c": [-1, -2, -3]}, {"d": [1, 1, 1]})},
            ]
        ),
        lambda: cudf.Series(
            [
                14.12302,
                97938.2,
                np.nan,
                0.0,
                -8.302014,
                np.nan,
                -112.2314,
            ]
        ).astype(cudf.Decimal64Dtype(7, 2)),
        lambda: cudf.DataFrame({"x": [1, 2, 3]}),
        lambda: cudf.DataFrame({"x": [1, 2, 3], "y": [1.0, None, 3.0]}),
        lambda: cudf.DataFrame(
            {"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]}, index=[1, None, 3]
        ),
        lambda: cudf.DataFrame(
            {"x": [1, 2, 3], "y": [1.0, None, 3.0]}, index=[1, None, 3]
        ),
        lambda: cudf.DataFrame(
            {"x": ["a", "bb", "ccc"], "y": [1.0, None, 3.0]},
            index=[1, None, 3],
        ),
        lambda: pd.Index([True, False] * 5),
        lambda: pd.CategoricalIndex(["a", "b", "a", "b"], ["a", "b", "c"]),
        lambda: (
            cudf.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": ["c", "e", "g"],
                    "d": [True, False, True],
                },
                index=cudf.MultiIndex.from_tuples(
                    [("i1", "i2"), ("i3", "i4"), ("i5", "i6")],
                    names=["foo", "bar"],
                ),
            )
        ),
        lambda: cudf.Index(
            cudf.date_range(start="2011-01-01", end="2012-01-01", periods=13)
        ),
        lambda: cudf.Index([1.2, 3.4, 5.6]),
        lambda: cudf.Series([1.2, 3.4, 5.6]),
        lambda: pd.IntervalIndex.from_breaks(range(10)),
        lambda: cudf.MultiIndex.from_tuples(
            [("i1", "i2"), ("i3", "i4"), ("i5", "i6")], names=["foo", "bar"]
        ),
        lambda: cudf.RangeIndex(10),
        lambda: cudf.DataFrame(
            {"a": list(range(13)), "b": [float(x) for x in range(13)]},
            index=cudf.Index(
                cudf.date_range(
                    start="2011-01-01", end="2012-01-01", periods=13
                )
            ),
        ),
        lambda: cudf.Series(
            list(range(13)),
            index=cudf.Index(
                cudf.date_range(
                    start="2011-01-01", end="2012-01-01", periods=13
                )
            ),
        ),
        lambda: cudf.TimedeltaIndex(
            [1132223, 2023232, 342234324, 4234324],
            dtype="timedelta64[ns]",
            name="foo",
        ),
        lambda: cudf.Index(
            [
                "y7ssMP1PWJ",
                "rZDLbzIQsX",
                "NrPwYMsxNw",
                "4zja1Vw9Rq",
                "Y9TNDhjXgR",
                "Ryjt7up2hT",
                "dxYKtRGHkb",
                "nMCWj5yhMu",
                "Rt7S362FNX",
                "OGbssOJLUI",
            ]
        ),
    ],
    ids=itertools.count(),
)
@pytest.mark.parametrize("to_host", [True, False])
def test_serialize(df, to_host):
    """This should hopefully replace all functions below"""
    a = df()
    if "cudf" not in type(a).__module__:
        a = cudf.from_pandas(a)
    if to_host:
        header, frames = a.host_serialize()
    else:
        header, frames = a.device_serialize()
    msgpack.dumps(header)  # ensure that header is msgpack serializable
    ndevice = 0
    for frame in frames:
        if hasattr(frame, "__cuda_array_interface__"):
            ndevice += 1
    # Indices etc. will not be DeviceNDArray
    # but data should be...
    if to_host:
        assert ndevice == 0
    elif hasattr(df, "_cols"):
        assert ndevice >= len(df._data)
    else:
        # If there are frames, something should be on the device
        assert ndevice > 0 or not frames

    typ = type(a)
    b = typ.deserialize(header, frames)
    assert_eq(a, b)


def test_serialize_dtype_error_checking():
    dtype = cudf.IntervalDtype("float", "right")
    header, frames = dtype.serialize()
    with pytest.raises(AssertionError):
        # Invalid number of frames
        type(dtype).deserialize(header, [None] * (header["frame_count"] + 1))
    with pytest.raises(AssertionError):
        # mismatching class
        cudf.StructDtype.deserialize(header, frames)


def test_serialize_dataframe():
    df = cudf.DataFrame()
    df["a"] = np.arange(100)
    df["b"] = np.arange(100, dtype=np.float32)
    df["c"] = pd.Categorical(
        ["a", "b", "c", "_", "_"] * 20, categories=["a", "b", "c"]
    )
    outdf = cudf.DataFrame.deserialize(*df.serialize())
    assert_eq(df, outdf)


def test_serialize_dataframe_with_index():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {
            "a": np.arange(100),
            "b": rng.random(100),
            "c": pd.Categorical(
                ["a", "b", "c", "_", "_"] * 20, categories=["a", "b", "c"]
            ),
        }
    )
    df = df.sort_values("b")
    outdf = cudf.DataFrame.deserialize(*df.serialize())
    assert_eq(df, outdf)


def test_serialize_series():
    sr = cudf.Series(np.arange(100))
    outsr = cudf.Series.deserialize(*sr.serialize())
    assert_eq(sr, outsr)


def test_serialize_range_index():
    index = cudf.core.index.RangeIndex(10, 20)
    outindex = cudf.core.index.RangeIndex.deserialize(*index.serialize())
    assert_eq(index, outindex)


def test_serialize_generic_index():
    index = cudf.core.index.Index(cudf.Series(np.arange(10)))
    outindex = cudf.core.index.Index.deserialize(*index.serialize())
    assert_eq(index, outindex)


def test_serialize_multi_index():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": [4, 17, 4, 9, 5],
            "b": [1, 4, 4, 3, 2],
            "x": rng.normal(size=5),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)
    gdg = gdf.groupby(["a", "b"]).sum()
    multiindex = gdg.index
    outindex = cudf.core.multiindex.MultiIndex.deserialize(
        *multiindex.serialize()
    )
    assert_eq(multiindex, outindex)


def test_serialize_masked_series():
    nelem = 50
    rng = np.random.default_rng(seed=0)
    data = rng.random(nelem)
    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]
    null_count = utils.count_zero(bitmask)
    assert null_count >= 0
    sr = cudf.Series.from_masked_array(data, mask, null_count=null_count)
    outsr = cudf.Series.deserialize(*sr.serialize())
    assert_eq(sr, outsr)


def test_serialize_groupby_df():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {
            "key_1": rng.integers(0, 20, 100),
            "key_2": rng.integers(0, 20, 100),
            "val": np.arange(100, dtype=np.float32),
        }
    )
    gb = df.groupby(["key_1", "key_2"], sort=True)
    outgb = gb.deserialize(*gb.serialize())
    expect = gb.mean()
    got = outgb.mean()
    assert_eq(got.sort_index(), expect.sort_index())


def test_serialize_groupby_external():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame({"val": np.arange(100, dtype=np.float32)})
    gb = df.groupby(cudf.Series(rng.integers(0, 20, 100)))
    outgb = gb.deserialize(*gb.serialize())
    expect = gb.mean()
    got = outgb.mean()
    assert_eq(got.sort_index(), expect.sort_index())


def test_serialize_groupby_level():
    idx = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 2)], names=("a", "b"))
    pdf = pd.DataFrame({"c": [1, 2, 3], "d": [2, 3, 4]}, index=idx)
    df = cudf.from_pandas(pdf)
    gb = df.groupby(level="a")
    expect = gb.mean()
    outgb = gb.deserialize(*gb.serialize())
    got = outgb.mean()
    assert_eq(expect.sort_index(), got.sort_index())


def test_serialize_groupby_sr():
    rng = np.random.default_rng(seed=0)
    sr = cudf.Series(rng.integers(0, 20, 100))
    gb = sr.groupby(sr // 2)
    outgb = gb.deserialize(*gb.serialize())
    got = gb.mean()
    expect = outgb.mean()
    assert_eq(got.sort_index(), expect.sort_index())


def test_serialize_datetime():
    rng = np.random.default_rng(seed=0)
    # Make frame with datetime column
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=20), "y": rng.normal(size=20)}
    )
    ts = np.arange(0, len(df), dtype=np.dtype("datetime64[ms]"))
    df["timestamp"] = ts
    gdf = cudf.DataFrame.from_pandas(df)
    # (De)serialize roundtrip
    recreated = cudf.DataFrame.deserialize(*gdf.serialize())
    # Check
    assert_eq(recreated, df)


def test_serialize_string():
    rng = np.random.default_rng(seed=0)
    # Make frame with string column
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=5), "y": rng.normal(size=5)}
    )
    str_data = ["a", "bc", "def", "ghij", "klmno"]
    df["timestamp"] = str_data
    gdf = cudf.DataFrame.from_pandas(df)
    # (De)serialize roundtrip
    recreated = cudf.DataFrame.deserialize(*gdf.serialize())
    # Check
    assert_eq(recreated, df)


@pytest.mark.parametrize(
    "frames",
    [
        (cudf.Series([], dtype="str"), pd.Series([], dtype="str")),
        (cudf.DataFrame(), pd.DataFrame()),
        (cudf.DataFrame([]), pd.DataFrame([])),
        (cudf.DataFrame({}), pd.DataFrame({})),
        (cudf.DataFrame([1]).head(0), pd.DataFrame([1]).head(0)),
        (cudf.DataFrame({"a": []}), pd.DataFrame({"a": []})),
        (
            cudf.DataFrame({"a": ["a"]}).head(0),
            pd.DataFrame({"a": ["a"]}).head(0),
        ),
        (
            cudf.DataFrame({"a": [1.0]}).head(0),
            pd.DataFrame({"a": [1.0]}).head(0),
        ),
    ],
)
def test_serialize_empty(frames):
    gdf, pdf = frames

    typ = type(gdf)
    res = typ.deserialize(*gdf.serialize())
    assert_eq(res, gdf)


def test_serialize_all_null_string():
    data = [None, None, None, None, None]
    pd_series = pd.Series(data, dtype="str")
    gd_series = cudf.Series(data, dtype="str")

    recreated = cudf.Series.deserialize(*gd_series.serialize())
    assert_eq(recreated, pd_series)


def test_serialize_named_series():
    gdf = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [5, 1, 2, 5]})

    ser = gdf["b"]
    recreated = cudf.Series.deserialize(*ser.serialize())
    assert_eq(recreated, ser)


def test_serialize_seriesgroupby():
    gdf = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [5, 1, 2, 5]})

    gb = gdf.groupby(["a"]).b
    recreated = gb.__class__.deserialize(*gb.serialize())
    assert_eq(recreated.sum(), gb.sum())


def test_serialize_seriesresampler():
    index = cudf.date_range(start="2001-01-01", periods=10, freq="1min")
    sr = cudf.Series(range(10), index=index)
    re_sampler = sr.resample("3min")
    actual = re_sampler.sum()
    recreated = re_sampler.__class__.deserialize(*re_sampler.serialize())
    expected = recreated.sum()

    assert_eq(actual, expected)


def test_serialize_string_check_buffer_sizes():
    df = cudf.DataFrame({"a": ["a", "b", "cd", None]})
    expect = df.memory_usage(deep=True).loc["a"]
    header, frames = df.serialize()
    got = sum(b.nbytes for b in frames)
    assert expect == got


def test_deserialize_cudf_23_12(datadir):
    fname = datadir / "pkl" / "stringColumnWithRangeIndex_cudf_23.12.pkl"

    expected = cudf.DataFrame({"a": ["hi", "hello", "world", None]})
    with open(fname, "rb") as f:
        actual = pickle.load(f)

    assert_eq(expected, actual)


def test_serialize_sliced_string():
    # https://github.com/rapidsai/cudf/issues/7735
    data = ["hi", "hello", None]
    pd_series = pd.Series(data, dtype=pd.StringDtype())
    gd_series = cudf.Series(data, dtype="str")
    sliced = gd_series[0:3]
    serialized_gd_series = gd_series.serialize()
    serialized_sliced = sliced.serialize()

    # validate frames are equal or not
    # because both should be identical
    for i in range(3):
        assert_eq(
            serialized_gd_series[1][i].memoryview(),
            serialized_sliced[1][i].memoryview(),
        )

    recreated = cudf.Series.deserialize(*sliced.serialize())
    assert_eq(recreated.to_pandas(nullable=True), pd_series)


@pytest.mark.parametrize(
    "columns",
    [
        cudf.RangeIndex(2),
        cudf.Index([1, 2], dtype="int8"),
        cudf.MultiIndex(
            levels=[["a", "b"], [1, 2]], codes=[[0, 1], [0, 1]], names=["a", 0]
        ),
    ],
)
def test_serialize_column_types_preserved(columns):
    expected = cudf.DataFrame([[10, 11]], columns=columns)
    result = cudf.DataFrame.deserialize(*expected.serialize())
    assert_eq(result, expected)
