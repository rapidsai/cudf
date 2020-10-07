# Copyright (c) 2020, NVIDIA CORPORATION.

import os
from contextlib import contextmanager
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pytest

import cudf
from cudf.tests.utils import assert_eq

s3fs = pytest.importorskip("s3fs")
boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
httpretty = pytest.importorskip("httpretty")


@contextmanager
def ensure_safe_environment_variables():
    """
    Get a context manager to safely set environment variables
    All changes will be undone on close, hence environment variables set
    within this contextmanager will neither persist nor change global state.
    """
    saved_environ = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)


@contextmanager
def s3_context(bucket, files=None):
    if files is None:
        files = {}
    with ensure_safe_environment_variables():
        # temporary workaround as moto fails for botocore >= 1.11 otherwise,
        # see https://github.com/spulec/moto/issues/1924 & 1952
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "foobar_key")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foobar_secret")

        with moto.mock_s3():
            client = boto3.client("s3")
            client.create_bucket(Bucket=bucket, ACL="public-read-write")
            for f, data in files.items():
                client.put_object(Bucket=bucket, Key=f, Body=data)

            yield s3fs.S3FileSystem(anon=True)

            for f, data in files.items():
                try:

                    client.delete_object(Bucket=bucket, Key=f)
                except Exception:
                    pass
                finally:
                    httpretty.HTTPretty.disable()
                    httpretty.HTTPretty.reset()


@pytest.fixture
def pdf(scope="module"):
    df = pd.DataFrame()
    df["Integer"] = np.array([2345, 11987, 9027, 9027])
    df["Float"] = np.array([9.001, 8.343, 6, 2.781])
    df["Integer2"] = np.array([2345, 106, 2088, 789277])
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


def test_read_csv(pdf):
    # Write to buffer
    fname = "test_csv_reader.csv"
    bname = "csv"
    buffer = pdf.to_csv(index=False)
    with s3_context(bname, {fname: buffer}):
        got = cudf.read_csv("s3://{}/{}".format(bname, fname))

    assert_eq(pdf, got)


@pytest.mark.parametrize("chunksize", [None, 3])
def test_write_csv(pdf, chunksize):
    # Write to buffer
    fname = "test_csv_writer.csv"
    bname = "csv"
    gdf = cudf.from_pandas(pdf)
    with s3_context(bname) as s3fs:
        gdf.to_csv(
            "s3://{}/{}".format(bname, fname), index=False, chunksize=chunksize
        )
        assert s3fs.exists("s3://{}/{}".format(bname, fname))

        got = pd.read_csv("s3://{}/{}".format(bname, fname))

    assert_eq(pdf, got)


def test_read_parquet(pdf):
    fname = "test_parquet_reader.parquet"
    bname = "parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(bname, {fname: buffer}):
        got = cudf.read_parquet("s3://{}/{}".format(bname, fname))

    assert_eq(pdf, got)


def test_write_parquet(pdf):
    fname = "test_parquet_writer.parquet"
    bname = "parquet"
    gdf = cudf.from_pandas(pdf)
    with s3_context(bname) as s3fs:
        gdf.to_parquet("s3://{}/{}".format(bname, fname))
        assert s3fs.exists("s3://{}/{}".format(bname, fname))

        got = pd.read_parquet("s3://{}/{}".format(bname, fname))

    assert_eq(pdf, got)


def test_read_json():
    fname = "test_json_reader.json"
    bname = "json"
    buffer = (
        b'{"amount": 100, "name": "Alice"}\n'
        b'{"amount": 200, "name": "Bob"}\n'
        b'{"amount": 300, "name": "Charlie"}\n'
        b'{"amount": 400, "name": "Dennis"}\n'
    )

    with s3_context(bname, {fname: buffer}):
        got = cudf.read_json(
            "s3://{}/{}".format(bname, fname),
            engine="cudf",
            orient="records",
            lines=True,
        )

    expect = pd.read_json(buffer, lines=True)
    assert_eq(expect, got)


def test_read_orc(datadir):
    source_file = str(datadir / "orc" / "TestOrcFile.testSnappy.orc")
    fname = "test_orc_reader.orc"
    bname = "orc"
    expect = pa.orc.ORCFile(source_file).read().to_pandas()

    with open(source_file, "rb") as f:
        buffer = f.read()

    with s3_context(bname, {fname: buffer}):
        got = cudf.read_orc("s3://{}/{}".format(bname, fname))

    assert_eq(expect, got)


def test_write_orc(pdf):
    fname = "test_orc_writer.orc"
    bname = "orc"
    gdf = cudf.from_pandas(pdf)
    with s3_context(bname) as s3fs:
        gdf.to_orc("s3://{}/{}".format(bname, fname))
        assert s3fs.exists("s3://{}/{}".format(bname, fname))

        with s3fs.open("s3://{}/{}".format(bname, fname)) as f:
            got = pa.orc.ORCFile(f).read().to_pandas()

    assert_eq(pdf, got)
