# Copyright (c) 2018, NVIDIA CORPORATION.

import msgpack
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests import utils
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "df",
    [
        lambda: cudf.Series([1, 2, 3]),
        lambda: cudf.Series([1, 2, 3], index=[4, 5, 6]),
        lambda: cudf.Series([1, None, 3]),
        lambda: cudf.Series([1, 2, 3], index=[4, 5, None]),
        lambda: cudf.Series(["a", "bb", "ccc"]),
        lambda: cudf.Series(["a", None, "ccc"]),
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
        pd.util.testing.makeTimeDataFrame,
        pd.util.testing.makeMixedDataFrame,
        pd.util.testing.makeTimeDataFrame,
        # pd.util.testing.makeMissingDataframe, # Problem in distributed
        # pd.util.testing.makeMultiIndex, # Indices not serialized on device
    ],
)
def test_serialize(df):
    """ This should hopefully replace all functions below """
    a = df()
    if "cudf" not in type(a).__module__:
        a = cudf.from_pandas(a)
    header, frames = a.serialize()
    msgpack.dumps(header)  # ensure that header is msgpack serializable
    ndevice = 0
    for frame in frames:
        if not isinstance(frame, (bytes, memoryview)):
            ndevice += 1
    # Indices etc. will not be DeviceNDArray
    # but data should be...
    if hasattr(df, "_cols"):
        assert ndevice >= len(df._data)
    else:
        assert ndevice > 0

    typ = type(a)
    b = typ.deserialize(header, frames)
    assert_eq(a, b)


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
    df = cudf.DataFrame()
    df["a"] = np.arange(100)
    df["b"] = np.random.random(100)
    df["c"] = pd.Categorical(
        ["a", "b", "c", "_", "_"] * 20, categories=["a", "b", "c"]
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
    index = cudf.core.index.GenericIndex(cudf.Series(np.arange(10)))
    outindex = cudf.core.index.GenericIndex.deserialize(*index.serialize())
    assert_eq(index, outindex)


def test_serialize_multi_index():
    pdf = pd.DataFrame(
        {
            "a": [4, 17, 4, 9, 5],
            "b": [1, 4, 4, 3, 2],
            "x": np.random.normal(size=5),
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
    data = np.random.random(nelem)
    mask = utils.random_bitmask(nelem)
    bitmask = utils.expand_bits_to_bytes(mask)[:nelem]
    null_count = utils.count_zero(bitmask)
    assert null_count >= 0
    sr = cudf.Series.from_masked_array(data, mask, null_count=null_count)
    outsr = cudf.Series.deserialize(*sr.serialize())
    assert_eq(sr, outsr)


def test_serialize_groupby_df():
    df = cudf.DataFrame()
    df["key_1"] = np.random.randint(0, 20, 100)
    df["key_2"] = np.random.randint(0, 20, 100)
    df["val"] = np.arange(100, dtype=np.float32)
    gb = df.groupby(["key_1", "key_2"])
    outgb = gb.deserialize(*gb.serialize())
    expect = gb.mean()
    got = outgb.mean()
    assert_eq(got, expect)


def test_serialize_groupby_external():
    df = cudf.DataFrame()
    df["val"] = np.arange(100, dtype=np.float32)
    gb = df.groupby(cudf.Series(np.random.randint(0, 20, 100)))
    outgb = gb.deserialize(*gb.serialize())
    expect = gb.mean()
    got = outgb.mean()
    assert_eq(got, expect)


def test_serialize_groupby_level():
    idx = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 2)], names=("a", "b"))
    pdf = pd.DataFrame({"c": [1, 2, 3], "d": [2, 3, 4]}, index=idx)
    df = cudf.from_pandas(pdf)
    gb = df.groupby(level="a")
    expect = gb.mean()
    outgb = gb.deserialize(*gb.serialize())
    got = outgb.mean()
    assert_eq(expect, got)


def test_serialize_groupby_sr():
    sr = cudf.Series(np.random.randint(0, 20, 100))
    gb = sr.groupby(sr // 2)
    outgb = gb.deserialize(*gb.serialize())
    got = gb.mean()
    expect = outgb.mean()
    assert_eq(got, expect)


def test_serialize_datetime():
    # Make frame with datetime column
    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=20), "y": np.random.normal(size=20)}
    )
    ts = np.arange(0, len(df), dtype=np.dtype("datetime64[ms]"))
    df["timestamp"] = ts
    gdf = cudf.DataFrame.from_pandas(df)
    # (De)serialize roundtrip
    recreated = cudf.DataFrame.deserialize(*gdf.serialize())
    # Check
    assert_eq(recreated, df)


def test_serialize_string():
    # Make frame with string column
    df = pd.DataFrame(
        {"x": np.random.randint(0, 5, size=5), "y": np.random.normal(size=5)}
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
        (cudf.DataFrame([]), pd.DataFrame([])),
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
