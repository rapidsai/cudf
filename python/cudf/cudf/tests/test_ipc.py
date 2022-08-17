# Copyright (c) 2022, NVIDIA CORPORATION.

import multiprocessing
import os
import tempfile
import pytest
import cupy as cp

import cudf
from cudf.testing._utils import assert_eq


def import_ipc(message, tmpfile) -> None:
    df = cudf.DataFrame.import_ipc(message)
    df.to_csv(tmpfile, index=False)


def check_roundtrip(df: cudf.DataFrame) -> None:
    msg = df.export_ipc()
    mctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "result.csv")
        p = mctx.Process(target=import_ipc, args=(msg, tmpfile))
        p.start()
        p.join()
        assert p.exitcode == 0
        res = cudf.read_csv(tmpfile)
        assert_eq(df, res)


def test_ipc_simple() -> None:
    df = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
    check_roundtrip(df)

    df = cudf.DataFrame(cp.arange(0, 16).reshape(4, 4))
    with pytest.raises(TypeError):
        # export IPC uses the smae column meta as interop, which doesn't support integer
        # index. df.to_arrow() should fail as well.
        df.export_ipc()

    with pytest.raises(RuntimeError):
        # list is not supported yet.
        df = cudf.DataFrame({"a": [cp.arange(0, 4)] * 4})
        df.export_ipc()
        check_roundtrip(df)


def test_ipc_with_null_mask():
    x = cp.arange(0.0, 10.0 * 10)
    y = x + 10.0
    z = y + 10.0
    x[1:3] = cp.nan
    y[:4] = cp.nan
    df = cudf.DataFrame({"x": x, "y": y, "z": z}, nan_as_null=True)
    mctx = multiprocessing.get_context("spawn")
    msg = df.export_ipc()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "result.csv")
        p = mctx.Process(target=import_ipc, args=(msg, tmpfile))
        p.start()
        p.join()
        assert p.exitcode == 0
        res = cudf.read_csv(tmpfile)
        assert_eq(df, res)
