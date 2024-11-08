# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import os
import socket
from contextlib import contextmanager
from io import BytesIO

import fsspec
import pandas as pd
import pytest

from dask.dataframe import assert_eq

import cudf

import dask_cudf
from dask_cudf.tests.utils import QUERY_PLANNING_ON

moto = pytest.importorskip("moto", minversion="3.1.6")
boto3 = pytest.importorskip("boto3")
s3fs = pytest.importorskip("s3fs")
ThreadedMotoServer = pytest.importorskip("moto.server").ThreadedMotoServer


@pytest.fixture(scope="session")
def endpoint_ip():
    return "127.0.0.1"


@pytest.fixture(scope="session")
def endpoint_port():
    # Return a free port per worker session.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


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
def s3_base(endpoint_ip, endpoint_port):
    """
    Fixture to set up moto server in separate process
    """
    with ensure_safe_environment_variables():
        # Fake aws credentials exported to prevent botocore looking for
        # system aws credentials, https://github.com/spulec/moto/issues/1793
        os.environ["AWS_ACCESS_KEY_ID"] = "foobar_key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foobar_secret"
        os.environ["S3FS_LOGGING_LEVEL"] = "DEBUG"
        os.environ["AWS_SECURITY_TOKEN"] = "foobar_security_token"
        os.environ["AWS_SESSION_TOKEN"] = "foobar_session_token"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

        # Launching moto in server mode, i.e., as a separate process
        # with an S3 endpoint on localhost

        endpoint_uri = f"http://{endpoint_ip}:{endpoint_port}/"

        server = ThreadedMotoServer(ip_address=endpoint_ip, port=endpoint_port)
        server.start()
        yield endpoint_uri
        server.stop()


@pytest.fixture()
def s3so(endpoint_ip, endpoint_port):
    """
    Returns s3 storage options to pass to fsspec
    """
    endpoint_uri = f"http://{endpoint_ip}:{endpoint_port}/"

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
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": [2.1, 2.2, 2.3, 2.4]})


def test_read_csv(s3_base, s3so):
    with s3_context(
        s3_base=s3_base, bucket="daskcsv", files={"a.csv": b"a,b\n1,2\n3,4\n"}
    ):
        df = dask_cudf.read_csv(
            "s3://daskcsv/*.csv", blocksize="50 B", storage_options=s3so
        )
        assert df.a.sum().compute() == 4


def test_read_parquet_open_file_options_raises():
    with pytest.raises(ValueError):
        dask_cudf.read_parquet(
            "s3://my/path",
            open_file_options={"precache_options": {"method": "parquet"}},
        )


@pytest.mark.parametrize(
    "filesystem",
    [
        pytest.param(
            "arrow",
            marks=pytest.mark.skipif(
                not QUERY_PLANNING_ON or not dask_cudf.backends.PYARROW_GE_15,
                reason="Not supported",
            ),
        ),
        "fsspec",
    ],
)
def test_read_parquet_filesystem(s3_base, s3so, pdf, filesystem):
    fname = "test_parquet_filesystem.parquet"
    bucket = "parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        path = f"s3://{bucket}/{fname}"
        if filesystem == "arrow":
            # This feature requires arrow >= 15
            pytest.importorskip("pyarrow", minversion="15.0.0")

            import pyarrow.fs as pa_fs

            df = dask_cudf.read_parquet(
                path,
                filesystem=pa_fs.S3FileSystem(
                    endpoint_override=s3so["client_kwargs"]["endpoint_url"],
                ),
            )
        else:
            df = dask_cudf.read_parquet(
                path,
                storage_options=s3so,
                filesystem=filesystem,
            )
        assert df.b.sum().compute() == 9
        assert isinstance(df._meta, cudf.DataFrame)
        assert isinstance(df.compute(), cudf.DataFrame)


def test_read_parquet_filesystem_explicit(s3_base, s3so, pdf):
    fname = "test_parquet_filesystem_explicit.parquet"
    bucket = "parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        path = f"s3://{bucket}/{fname}"
        fs = fsspec.core.get_fs_token_paths(
            path, mode="rb", storage_options=s3so
        )[0]
        df = dask_cudf.read_parquet(path, filesystem=fs)
        assert df.b.sum().compute() == 9


def test_read_parquet(s3_base, s3so, pdf):
    fname = "test_parquet_reader_dask.parquet"
    bucket = "parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got = dask_cudf.read_parquet(
            f"s3://{bucket}/{fname}",
            storage_options=s3so,
        )
        assert_eq(pdf, got)


def test_read_orc(s3_base, s3so, pdf):
    fname = "test_orc_reader_dask.orc"
    bucket = "orc"
    buffer = BytesIO()
    pdf.to_orc(path=buffer)
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got = dask_cudf.read_orc(
            f"s3://{bucket}/{fname}",
            storage_options=s3so,
        )
        assert_eq(pdf, got)
