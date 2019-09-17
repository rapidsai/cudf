import os
from contextlib import contextmanager

import pytest

import dask_cudf

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
def s3_context(bucket, files):
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


def test_read_csv():
    with s3_context("daskcsv", {"a.csv": b"a,b\n1,2\n3,4\n"}):
        df = dask_cudf.read_csv("s3://daskcsv/*.csv", chunksize="50 B")
        assert df.a.sum().compute() == 4
