import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from io import BytesIO

import pandas as pd
import pytest

import dask_cudf

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


def test_read_csv(s3_base, s3so):
    with s3_context(
        s3_base=s3_base, bucket="daskcsv", files={"a.csv": b"a,b\n1,2\n3,4\n"}
    ):
        df = dask_cudf.read_csv(
            "s3://daskcsv/*.csv", chunksize="50 B", storage_options=s3so
        )
        assert df.a.sum().compute() == 4


def test_read_parquet(s3_base, s3so):
    pdf = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2.1, 2.2, 2.3, 2.4]})
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(
        s3_base=s3_base, bucket="daskparquet", files={"file.parq": buffer}
    ):
        df = dask_cudf.read_parquet(
            "s3://daskparquet/*.parq", storage_options=s3so
        )
        assert df.a.sum().compute() == 10
        assert df.b.sum().compute() == 9
