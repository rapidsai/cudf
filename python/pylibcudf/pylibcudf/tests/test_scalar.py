# Copyright (c) 2024-2025, NVIDIA CORPORATION.
import datetime

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.fixture(scope="module")
def np():
    return pytest.importorskip("numpy")


@pytest.mark.parametrize(
    "val", [True, False, -1, 0, 1 - 1.0, 0.0, 1.52, "", "a1!"]
)
def test_from_py(val):
    result = plc.Scalar.from_py(val)
    expected = pa.scalar(val)
    assert plc.interop.to_arrow(result).equals(expected)


@pytest.mark.parametrize(
    "val", [datetime.datetime(2020, 1, 1), datetime.timedelta(1), [1], {1: 1}]
)
def test_from_py_notimplemented(val):
    with pytest.raises(NotImplementedError):
        plc.Scalar.from_py(val)


@pytest.mark.parametrize("val", [object, None])
def test_from_py_typeerror(val):
    with pytest.raises(TypeError):
        plc.Scalar.from_py(val)


@pytest.mark.parametrize(
    "np_type",
    [
        "bool_",
        "str_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_from_numpy(np, np_type):
    np_klass = getattr(np, np_type)
    np_val = np_klass("1" if np_type == "str_" else 1)
    result = plc.Scalar.from_numpy(np_val)
    expected = pa.scalar(np_val)
    assert plc.interop.to_arrow(result).equals(expected)


@pytest.mark.parametrize("np_type", ["datetime64", "timedelta64"])
def test_from_numpy_notimplemented(np, np_type):
    np_val = getattr(np, np_type)(1, "ns")
    with pytest.raises(NotImplementedError):
        plc.Scalar.from_numpy(np_val)


def test_from_numpy_typeerror(np):
    with pytest.raises(TypeError):
        plc.Scalar.from_numpy(np.void(5))
