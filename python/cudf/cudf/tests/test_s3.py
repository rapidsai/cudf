# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import os
import socket
from contextlib import contextmanager
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pytest
from fsspec.core import get_fs_token_paths

import cudf
from cudf.testing import assert_eq

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
        os.environ["AWS_ENDPOINT_URL"] = endpoint_uri

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


@pytest.fixture(
    params=[True, False],
    ids=["kvikio=ON", "kvikio=OFF"],
)
def kvikio_remote_io(request):
    with cudf.option_context("kvikio_remote_io", request.param):
        yield request.param


@pytest.fixture
def pdf(scope="module"):
    df = pd.DataFrame()
    df["Integer"] = np.array([2345, 11987, 9027, 9027])
    df["Float"] = np.array([9.001, 8.343, 6, 2.781])
    df["Integer2"] = np.array([2345, 106, 2088, 789277])
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


@pytest.fixture
def pdf_ext(scope="module"):
    size = 100
    df = pd.DataFrame()
    df["Integer"] = np.array([i for i in range(size)])
    df["List"] = [[i] for i in range(size)]
    df["Struct"] = [{"a": i} for i in range(size)]
    df["String"] = (["Alpha", "Beta", "Gamma", "Delta"] * (-(size // -4)))[
        :size
    ]
    return df


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
def test_read_csv(s3_base, s3so, pdf, bytes_per_thread):
    # Write to buffer
    fname = "test_csv_reader.csv"
    bucket = "csv"
    buffer = pdf.to_csv(index=False)

    # Use fsspec file object
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got = cudf.read_csv(
            f"s3://{bucket}/{fname}",
            storage_options=s3so,
            bytes_per_thread=bytes_per_thread,
        )
    assert_eq(pdf, got)


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
def test_read_csv_byte_range(s3_base, s3so, pdf, bytes_per_thread):
    # Write to buffer
    fname = "test_csv_reader_byte_range.csv"
    bucket = "csv"
    buffer = pdf.to_csv(index=False)

    # Use fsspec file object
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got = cudf.read_csv(
            f"s3://{bucket}/{fname}",
            storage_options=s3so,
            byte_range=(74, 73),
            bytes_per_thread=bytes_per_thread,
            header=None,
            names=["Integer", "Float", "Integer2", "String", "Boolean"],
        )

    assert_eq(pdf.iloc[-2:].reset_index(drop=True), got)


@pytest.mark.parametrize("chunksize", [None, 3])
def test_write_csv(s3_base, s3so, pdf, chunksize):
    # Write to buffer
    fname = "test_csv_writer.csv"
    bucket = "csv"
    gdf = cudf.from_pandas(pdf)
    with s3_context(s3_base=s3_base, bucket=bucket) as s3fs:
        gdf.to_csv(
            f"s3://{bucket}/{fname}",
            index=False,
            chunksize=chunksize,
            storage_options=s3so,
        )
        assert s3fs.exists(f"s3://{bucket}/{fname}")

        # TODO: Update to use `storage_options` from pandas v1.2.0
        got = pd.read_csv(s3fs.open(f"s3://{bucket}/{fname}"))

    assert_eq(pdf, got)


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
@pytest.mark.parametrize("columns", [None, ["Float", "String"]])
def test_read_parquet(
    s3_base,
    s3so,
    kvikio_remote_io,
    pdf,
    bytes_per_thread,
    columns,
):
    fname = "test_parquet_reader.parquet"
    bucket = "parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)

    # Check direct path handling
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got1 = cudf.read_parquet(
            f"s3://{bucket}/{fname}",
            storage_options=s3so,
            bytes_per_thread=bytes_per_thread,
            columns=columns,
        )
    expect = pdf[columns] if columns else pdf
    assert_eq(expect, got1)

    # Check fsspec file-object handling
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        fs = get_fs_token_paths(
            f"s3://{bucket}/{fname}", storage_options=s3so
        )[0]
        with fs.open(f"s3://{bucket}/{fname}", mode="rb") as f:
            got2 = cudf.read_parquet(
                f,
                bytes_per_thread=bytes_per_thread,
                columns=columns,
            )
    assert_eq(expect, got2)


@pytest.mark.parametrize("method", ["all", "parquet"])
@pytest.mark.parametrize("blocksize", [1024 * 1024, 1024])
def test_read_parquet_prefetch_options(
    s3_base,
    s3so,
    pdf,
    method,
    blocksize,
):
    bucket = "parquet"
    fname_1 = "test_parquet_reader_prefetch_options_1.parquet"
    buffer_1 = BytesIO()
    pdf.to_parquet(path=buffer_1)
    buffer_1.seek(0)

    fname_2 = "test_parquet_reader_prefetch_options_2.parquet"
    buffer_2 = BytesIO()
    pdf_2 = pdf.copy()
    pdf_2["Integer"] += 1
    pdf_2.to_parquet(path=buffer_2)
    buffer_2.seek(0)

    with s3_context(
        s3_base=s3_base,
        bucket=bucket,
        files={
            fname_1: buffer_1,
            fname_2: buffer_2,
        },
    ):
        got = cudf.read_parquet(
            [
                f"s3://{bucket}/{fname_1}",
                f"s3://{bucket}/{fname_2}",
            ],
            storage_options=s3so,
            prefetch_options={
                "method": method,
                "blocksize": blocksize,
            },
            columns=["String", "Integer"],
        )

    expect = pd.concat([pdf, pdf_2], ignore_index=True)[["String", "Integer"]]
    assert_eq(expect, got)


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
@pytest.mark.parametrize("columns", [None, ["List", "Struct"]])
@pytest.mark.parametrize("index", [None, "Integer"])
def test_read_parquet_ext(
    s3_base,
    s3so,
    pdf_ext,
    bytes_per_thread,
    columns,
    index,
):
    fname = "test_parquet_reader_ext.parquet"
    bucket = "parquet"
    buffer = BytesIO()

    if index:
        pdf_ext.set_index(index).to_parquet(path=buffer)
    else:
        pdf_ext.to_parquet(path=buffer)

    # Check direct path handling
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got1 = cudf.read_parquet(
            f"s3://{bucket}/{fname}",
            storage_options=s3so,
            bytes_per_thread=bytes_per_thread,
            columns=columns,
        )
    if index:
        expect = (
            pdf_ext.set_index(index)[columns]
            if columns
            else pdf_ext.set_index(index)
        )
    else:
        expect = pdf_ext[columns] if columns else pdf_ext
    assert_eq(expect, got1)


def test_read_parquet_filesystem(s3_base, s3so, pdf):
    fname = "data.0.parquet"
    # NOTE: Need a unique bucket name when a glob pattern
    # is used, otherwise fsspec seems to cache the bucket
    # contents, and later tests using the same bucket name
    # will fail.
    bucket = "test_read_parquet_filesystem"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    fs = get_fs_token_paths("s3://", mode="rb", storage_options=s3so)[0]
    with s3_context(
        s3_base=s3_base,
        bucket=bucket,
        files={fname: buffer},
    ):
        # Check that a glob pattern works
        path = f"s3://{bucket}/{'data.*.parquet'}"
        got = cudf.read_parquet(path, filesystem=fs)
    assert_eq(pdf, got)


def test_read_parquet_multi_file(s3_base, s3so, pdf):
    fname_1 = "test_parquet_reader_multi_file_1.parquet"
    buffer_1 = BytesIO()
    pdf.to_parquet(path=buffer_1)
    buffer_1.seek(0)

    fname_2 = "test_parquet_reader_multi_file_2.parquet"
    buffer_2 = BytesIO()
    pdf.to_parquet(path=buffer_2)
    buffer_2.seek(0)

    bucket = "parquet"
    with s3_context(
        s3_base=s3_base,
        bucket=bucket,
        files={
            fname_1: buffer_1,
            fname_2: buffer_2,
        },
    ):
        got = cudf.read_parquet(
            [
                f"s3://{bucket}/{fname_1}",
                f"s3://{bucket}/{fname_2}",
            ],
            storage_options=s3so,
        ).reset_index(drop=True)

    expect = pd.concat([pdf, pdf], ignore_index=True)
    assert_eq(expect, got)


def test_read_parquet_filters(s3_base, s3so, pdf_ext):
    fname = "test_parquet_reader_filters.parquet"
    bucket = "parquet"
    buffer = BytesIO()
    pdf_ext.to_parquet(path=buffer)
    buffer.seek(0)
    filters = [("String", "==", "Omega")]
    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got = cudf.read_parquet(
            f"s3://{bucket}/{fname}",
            storage_options=s3so,
            filters=filters,
        )

    # All row-groups should be filtered out
    assert_eq(pdf_ext.iloc[:0], got.reset_index(drop=True))


@pytest.mark.parametrize("partition_cols", [None, ["String"]])
def test_write_parquet(s3_base, s3so, pdf, partition_cols):
    fname_cudf = "test_parquet_writer_cudf"
    fname_pandas = "test_parquet_writer_pandas"
    bucket = "parquet"
    gdf = cudf.from_pandas(pdf)

    with s3_context(s3_base=s3_base, bucket=bucket) as s3fs:
        gdf.to_parquet(
            f"s3://{bucket}/{fname_cudf}",
            partition_cols=partition_cols,
            storage_options=s3so,
        )
        assert s3fs.exists(f"s3://{bucket}/{fname_cudf}")
        pdf.to_parquet(
            f"s3://{bucket}/{fname_pandas}",
            partition_cols=partition_cols,
            storage_options=s3so,
        )
        assert s3fs.exists(f"s3://{bucket}/{fname_pandas}")

        got = pd.read_parquet(
            f"s3://{bucket}/{fname_pandas}", storage_options=s3so
        )
        expect = cudf.read_parquet(
            f"s3://{bucket}/{fname_cudf}", storage_options=s3so
        )

    assert_eq(expect, got)


def test_read_json(s3_base, s3so):
    fname = "test_json_reader.json"
    bucket = "json"
    buffer = (
        '{"amount": 100, "name": "Alice"}\n'
        '{"amount": 200, "name": "Bob"}\n'
        '{"amount": 300, "name": "Charlie"}\n'
        '{"amount": 400, "name": "Dennis"}\n'
    )

    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got = cudf.read_json(
            f"s3://{bucket}/{fname}",
            engine="cudf",
            orient="records",
            lines=True,
            storage_options=s3so,
        )

    expect = pd.read_json(StringIO(buffer), lines=True)
    assert_eq(expect, got)


@pytest.mark.parametrize("columns", [None, ["string1"]])
def test_read_orc(s3_base, s3so, datadir, columns):
    source_file = str(datadir / "orc" / "TestOrcFile.testSnappy.orc")
    fname = "test_orc_reader.orc"
    bucket = "orc"
    expect = pd.read_orc(source_file)

    with open(source_file, "rb") as f:
        buffer = f.read()

    with s3_context(s3_base=s3_base, bucket=bucket, files={fname: buffer}):
        got = cudf.read_orc(
            f"s3://{bucket}/{fname}",
            columns=columns,
            storage_options=s3so,
        )

    if columns:
        expect = expect[columns]
    assert_eq(expect, got)


def test_write_orc(s3_base, s3so, pdf):
    fname = "test_orc_writer.orc"
    bucket = "orc"
    gdf = cudf.from_pandas(pdf)
    with s3_context(s3_base=s3_base, bucket=bucket) as s3fs:
        gdf.to_orc(f"s3://{bucket}/{fname}", storage_options=s3so)
        assert s3fs.exists(f"s3://{bucket}/{fname}")

        with s3fs.open(f"s3://{bucket}/{fname}") as f:
            got = pd.read_orc(f)

    assert_eq(pdf, got)


def test_write_chunked_parquet(s3_base, s3so):
    df1 = cudf.DataFrame({"b": [10, 11, 12], "a": [1, 2, 3]})
    df2 = cudf.DataFrame({"b": [20, 30, 50], "a": [3, 2, 1]})
    dirname = "chunked_writer_directory"
    bucket = "parquet"
    from cudf.io.parquet import ParquetDatasetWriter

    with s3_context(
        s3_base=s3_base, bucket=bucket, files={dirname: BytesIO()}
    ) as s3fs:
        with ParquetDatasetWriter(
            f"s3://{bucket}/{dirname}",
            partition_cols=["a"],
            storage_options=s3so,
        ) as cw:
            cw.write_table(df1)
            cw.write_table(df2)

        # TODO: Replace following workaround with:
        # expect = cudf.read_parquet(f"s3://{bucket}/{dirname}/",
        # storage_options=s3so)
        # after the following bug is fixed:
        # https://issues.apache.org/jira/browse/ARROW-16438

        dfs = []
        for folder in {"a=1", "a=2", "a=3"}:
            assert s3fs.exists(f"s3://{bucket}/{dirname}/{folder}")
            for file in s3fs.ls(f"s3://{bucket}/{dirname}/{folder}"):
                df = cudf.read_parquet("s3://" + file, storage_options=s3so)
                dfs.append(df)

        actual = cudf.concat(dfs).astype("int64")
        assert_eq(
            actual.sort_values(["b"]).reset_index(drop=True),
            cudf.concat([df1, df2]).sort_values(["b"]).reset_index(drop=True),
        )


def test_no_s3fs_on_cudf_import():
    import subprocess
    import sys

    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import cudf; import sys; print('pyarrow._s3fs' in sys.modules)",
        ],
        cwd="/",
    )
    assert output.strip() == b"False"
