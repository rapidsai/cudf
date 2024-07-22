# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import os
import socket
from contextlib import contextmanager
from io import BytesIO

import pandas as pd
import pyarrow.fs as pa_fs
import pytest

import dask_cudf

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


def test_read_csv(s3_base, s3so):
    with s3_context(
        s3_base=s3_base, bucket="daskcsv", files={"a.csv": b"a,b\n1,2\n3,4\n"}
    ):
        df = dask_cudf.read_csv(
            "s3://daskcsv/*.csv", blocksize="50 B", storage_options=s3so
        )
        assert df.a.sum().compute() == 4


@pytest.mark.parametrize(
    "open_file_options",
    [
        {"precache_options": {"method": None}},
        {"precache_options": {"method": "parquet"}},
        {"open_file_func": None},
    ],
)
def test_read_parquet_deprecated(s3_base, s3so, open_file_options):
    pdf = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2.1, 2.2, 2.3, 2.4]})
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(
        s3_base=s3_base, bucket="daskparquet", files={"file.parq": buffer}
    ):
        if "open_file_func" in open_file_options:
            fs = pa_fs.S3FileSystem(
                endpoint_override=s3so["client_kwargs"]["endpoint_url"],
            )
            open_file_options["open_file_func"] = fs.open_input_file
        df = dask_cudf.read_parquet(
            "s3://daskparquet/*.parq",
            storage_options=s3so,
            open_file_options=open_file_options,
        )
        with pytest.warns(FutureWarning):
            assert df.a.sum().compute() == 10
        with pytest.warns(FutureWarning):
            assert df.b.sum().compute() == 9


def test_read_parquet_remote(s3_base, s3so):
    pdf = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2.1, 2.2, 2.3, 2.4]})
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(
        s3_base=s3_base, bucket="daskparquet", files={"file.parq": buffer}
    ):
        df = dask_cudf.read_parquet(
            "s3://daskparquet/*.parq",
            storage_options=s3so,
        )
        assert df.a.sum().compute() == 10
        assert df.b.sum().compute() == 9
