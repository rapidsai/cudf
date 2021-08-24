# Copyright (c) 2020, NVIDIA CORPORATION.

import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pytest

import cudf
from cudf.testing._utils import assert_eq

moto = pytest.importorskip("moto", minversion="1.3.14")
boto3 = pytest.importorskip("boto3")
requests = pytest.importorskip("requests")
s3fs = pytest.importorskip("s3fs")


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


@pytest.fixture(scope="session")
def s3_base(worker_id):
    """
    Fixture to set up moto server in separate process
    """
    with ensure_safe_environment_variables():
        # Fake aws credentials exported to prevent botocore looking for
        # system aws credentials, https://github.com/spulec/moto/issues/1793
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "foobar_key")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foobar_secret")

        # Launching moto in server mode, i.e., as a separate process
        # with an S3 endpoint on localhost

        endpoint_port = (
            5000
            if worker_id == "master"
            else 5550 + int(worker_id.lstrip("gw"))
        )
        endpoint_uri = f"http://127.0.0.1:{endpoint_port}/"

        proc = subprocess.Popen(
            shlex.split(f"moto_server s3 -p {endpoint_port}"),
        )

        timeout = 5
        while timeout > 0:
            try:
                # OK to go once server is accepting connections
                r = requests.get(endpoint_uri)
                if r.ok:
                    break
            except Exception:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield endpoint_uri

        proc.terminate()
        proc.wait()


@pytest.fixture()
def s3so(worker_id):
    """
    Returns s3 storage options to pass to fsspec
    """
    endpoint_port = (
        5000 if worker_id == "master" else 5550 + int(worker_id.lstrip("gw"))
    )
    endpoint_uri = f"http://127.0.0.1:{endpoint_port}/"

    return {"client_kwargs": {"endpoint_url": endpoint_uri}}


@contextmanager
def s3_context(s3_base, bucket, files=None):
    if files is None:
        files = {}
    with ensure_safe_environment_variables():
        client = boto3.client("s3", endpoint_url=s3_base)
        client.create_bucket(Bucket=bucket, ACL="public-read-write")
        for f, data in files.items():
            client.put_object(Bucket=bucket, Key=f, Body=data)

        yield s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_base})

        for f, data in files.items():
            try:
                client.delete_object(Bucket=bucket, Key=f)
            except Exception:
                pass


@pytest.fixture
def pdf(scope="module"):
    df = pd.DataFrame()
    df["Integer"] = np.array([2345, 11987, 9027, 9027])
    df["Float"] = np.array([9.001, 8.343, 6, 2.781])
    df["Integer2"] = np.array([2345, 106, 2088, 789277])
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


def test_read_csv(s3_base, s3so, pdf):
    # Write to buffer
    fname = "test_csv_reader.csv"
    bname = "csv"
    buffer = pdf.to_csv(index=False)
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_csv(
            "s3://{}/{}".format(bname, fname), storage_options=s3so
        )

    assert_eq(pdf, got)


@pytest.mark.parametrize("chunksize", [None, 3])
def test_write_csv(s3_base, s3so, pdf, chunksize):
    # Write to buffer
    fname = "test_csv_writer.csv"
    bname = "csv"
    gdf = cudf.from_pandas(pdf)
    with s3_context(s3_base=s3_base, bucket=bname) as s3fs:
        gdf.to_csv(
            "s3://{}/{}".format(bname, fname),
            index=False,
            chunksize=chunksize,
            storage_options=s3so,
        )
        assert s3fs.exists("s3://{}/{}".format(bname, fname))

        # TODO: Update to use `storage_options` from pandas v1.2.0
        got = pd.read_csv(s3fs.open("s3://{}/{}".format(bname, fname)))

    assert_eq(pdf, got)


def test_read_parquet(s3_base, s3so, pdf):
    fname = "test_parquet_reader.parquet"
    bname = "parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_parquet(
            "s3://{}/{}".format(bname, fname), storage_options=s3so
        )

    assert_eq(pdf, got)


def test_write_parquet(s3_base, s3so, pdf):
    fname = "test_parquet_writer.parquet"
    bname = "parquet"
    gdf = cudf.from_pandas(pdf)
    with s3_context(s3_base=s3_base, bucket=bname) as s3fs:
        gdf.to_parquet("s3://{}/{}".format(bname, fname), storage_options=s3so)
        assert s3fs.exists("s3://{}/{}".format(bname, fname))

        got = pd.read_parquet(s3fs.open("s3://{}/{}".format(bname, fname)))

    assert_eq(pdf, got)


def test_read_json(s3_base, s3so):
    fname = "test_json_reader.json"
    bname = "json"
    buffer = (
        b'{"amount": 100, "name": "Alice"}\n'
        b'{"amount": 200, "name": "Bob"}\n'
        b'{"amount": 300, "name": "Charlie"}\n'
        b'{"amount": 400, "name": "Dennis"}\n'
    )

    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_json(
            "s3://{}/{}".format(bname, fname),
            engine="cudf",
            orient="records",
            lines=True,
            storage_options=s3so,
        )

    expect = pd.read_json(buffer, lines=True)
    assert_eq(expect, got)


def test_read_orc(s3_base, s3so, datadir):
    source_file = str(datadir / "orc" / "TestOrcFile.testSnappy.orc")
    fname = "test_orc_reader.orc"
    bname = "orc"
    expect = pa.orc.ORCFile(source_file).read().to_pandas()

    with open(source_file, "rb") as f:
        buffer = f.read()

    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_orc(
            "s3://{}/{}".format(bname, fname), storage_options=s3so
        )

    assert_eq(expect, got)


def test_write_orc(s3_base, s3so, pdf):
    fname = "test_orc_writer.orc"
    bname = "orc"
    gdf = cudf.from_pandas(pdf)
    with s3_context(s3_base=s3_base, bucket=bname) as s3fs:
        gdf.to_orc("s3://{}/{}".format(bname, fname), storage_options=s3so)
        assert s3fs.exists("s3://{}/{}".format(bname, fname))

        with s3fs.open("s3://{}/{}".format(bname, fname)) as f:
            got = pa.orc.ORCFile(f).read().to_pandas()

    assert_eq(pdf, got)
