# Copyright (c) 2022, NVIDIA CORPORATION.

import multiprocessing
import os
import tempfile
import time

import cupy as cp

import cudf
from cudf.testing._utils import assert_eq


def import_ipc(message, tmpfile) -> None:
    print("getpid:", os.getpid())
    time.sleep(8)

    df = cudf.DataFrame.import_ipc(message)
    df.to_csv(tmpfile, index=False)


def test_ipc_simple():
    df = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
    print(df)
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


def test_ipc_with_null_mask():
    x = cp.arange(0.0, 9.0)
    y = cp.arange(10.0, 19.0)
    x[1:3] = cp.nan
    y[:4] = cp.nan
    df = cudf.DataFrame({"x": x, "y": y}, nan_as_null=True)
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
