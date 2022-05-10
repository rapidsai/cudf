# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pa_fs
import pyarrow.orc
import pytest
from fsspec.core import get_fs_token_paths

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
    bname = "csv"
    buffer = pdf.to_csv(index=False)

    # Use fsspec file object
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_csv(
            f"s3://{bname}/{fname}",
            storage_options=s3so,
            bytes_per_thread=bytes_per_thread,
            use_python_file_object=False,
        )
    assert_eq(pdf, got)

    # Use Arrow PythonFile object
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_csv(
            f"s3://{bname}/{fname}",
            storage_options=s3so,
            bytes_per_thread=bytes_per_thread,
            use_python_file_object=True,
        )
    assert_eq(pdf, got)


def test_read_csv_arrow_nativefile(s3_base, s3so, pdf):
    # Write to buffer
    fname = "test_csv_reader_arrow_nativefile.csv"
    bname = "csv"
    buffer = pdf.to_csv(index=False)
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        fs = pa_fs.S3FileSystem(
            endpoint_override=s3so["client_kwargs"]["endpoint_url"],
        )
        with fs.open_input_file(f"{bname}/{fname}") as fil:
            got = cudf.read_csv(fil)

    assert_eq(pdf, got)


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
@pytest.mark.parametrize("use_python_file_object", [True, False])
def test_read_csv_byte_range(
    s3_base, s3so, pdf, bytes_per_thread, use_python_file_object
):
    # Write to buffer
    fname = "test_csv_reader_byte_range.csv"
    bname = "csv"
    buffer = pdf.to_csv(index=False)

    # Use fsspec file object
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_csv(
            f"s3://{bname}/{fname}",
            storage_options=s3so,
            byte_range=(74, 73),
            bytes_per_thread=bytes_per_thread,
            header=None,
            names=["Integer", "Float", "Integer2", "String", "Boolean"],
            use_python_file_object=use_python_file_object,
        )

    assert_eq(pdf.iloc[-2:].reset_index(drop=True), got)


@pytest.mark.parametrize("chunksize", [None, 3])
def test_write_csv(s3_base, s3so, pdf, chunksize):
    # Write to buffer
    fname = "test_csv_writer.csv"
    bname = "csv"
    gdf = cudf.from_pandas(pdf)
    with s3_context(s3_base=s3_base, bucket=bname) as s3fs:
        gdf.to_csv(
            f"s3://{bname}/{fname}",
            index=False,
            chunksize=chunksize,
            storage_options=s3so,
        )
        assert s3fs.exists(f"s3://{bname}/{fname}")

        # TODO: Update to use `storage_options` from pandas v1.2.0
        got = pd.read_csv(s3fs.open(f"s3://{bname}/{fname}"))

    assert_eq(pdf, got)


@pytest.mark.parametrize("bytes_per_thread", [32, 1024])
@pytest.mark.parametrize("columns", [None, ["Float", "String"]])
@pytest.mark.parametrize("precache", [None, "parquet"])
@pytest.mark.parametrize("use_python_file_object", [True, False])
def test_read_parquet(
    s3_base,
    s3so,
    pdf,
    bytes_per_thread,
    columns,
    precache,
    use_python_file_object,
):
    fname = "test_parquet_reader.parquet"
    bname = "parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)

    # Check direct path handling
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got1 = cudf.read_parquet(
            f"s3://{bname}/{fname}",
            open_file_options=(
                {"precache_options": {"method": precache}}
                if use_python_file_object
                else None
            ),
            storage_options=s3so,
            bytes_per_thread=bytes_per_thread,
            columns=columns,
            use_python_file_object=use_python_file_object,
        )
    expect = pdf[columns] if columns else pdf
    assert_eq(expect, got1)

    # Check fsspec file-object handling
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        fs = get_fs_token_paths(f"s3://{bname}/{fname}", storage_options=s3so)[
            0
        ]
        with fs.open(f"s3://{bname}/{fname}", mode="rb") as f:
            got2 = cudf.read_parquet(
                f,
                bytes_per_thread=bytes_per_thread,
                columns=columns,
                use_python_file_object=use_python_file_object,
            )
    assert_eq(expect, got2)


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
    bname = "parquet"
    buffer = BytesIO()

    if index:
        pdf_ext.set_index(index).to_parquet(path=buffer)
    else:
        pdf_ext.to_parquet(path=buffer)

    # Check direct path handling
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got1 = cudf.read_parquet(
            f"s3://{bname}/{fname}",
            storage_options=s3so,
            bytes_per_thread=bytes_per_thread,
            footer_sample_size=3200,
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


@pytest.mark.parametrize("columns", [None, ["Float", "String"]])
def test_read_parquet_arrow_nativefile(s3_base, s3so, pdf, columns):
    # Write to buffer
    fname = "test_parquet_reader_arrow_nativefile.parquet"
    bname = "parquet"
    buffer = BytesIO()
    pdf.to_parquet(path=buffer)
    buffer.seek(0)
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        fs = pa_fs.S3FileSystem(
            endpoint_override=s3so["client_kwargs"]["endpoint_url"],
        )
        with fs.open_input_file(f"{bname}/{fname}") as fil:
            got = cudf.read_parquet(fil, columns=columns)

    expect = pdf[columns] if columns else pdf
    assert_eq(expect, got)


@pytest.mark.parametrize("precache", [None, "parquet"])
def test_read_parquet_filters(s3_base, s3so, pdf_ext, precache):
    fname = "test_parquet_reader_filters.parquet"
    bname = "parquet"
    buffer = BytesIO()
    pdf_ext.to_parquet(path=buffer)
    buffer.seek(0)
    filters = [("String", "==", "Omega")]
    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_parquet(
            f"s3://{bname}/{fname}",
            storage_options=s3so,
            filters=filters,
            open_file_options={"precache_options": {"method": precache}},
        )

    # All row-groups should be filtered out
    assert_eq(pdf_ext.iloc[:0], got.reset_index(drop=True))


@pytest.mark.parametrize("partition_cols", [None, ["String"]])
def test_write_parquet(s3_base, s3so, pdf, partition_cols):
    fname = "test_parquet_writer.parquet"
    bname = "parquet"
    gdf = cudf.from_pandas(pdf)
    with s3_context(s3_base=s3_base, bucket=bname) as s3fs:
        gdf.to_parquet(
            f"s3://{bname}/{fname}",
            partition_cols=partition_cols,
            storage_options=s3so,
        )
        assert s3fs.exists(f"s3://{bname}/{fname}")

        got = pd.read_parquet(s3fs.open(f"s3://{bname}/{fname}"))

    assert_eq(pdf, got)


def test_read_json(s3_base, s3so):
    fname = "test_json_reader.json"
    bname = "json"
    # TODO: After following bug is fixed switch
    # back to using bytes:
    # https://github.com/pandas-dev/pandas/issues/46935

    # buffer = (
    #     b'{"amount": 100, "name": "Alice"}\n'
    #     b'{"amount": 200, "name": "Bob"}\n'
    #     b'{"amount": 300, "name": "Charlie"}\n'
    #     b'{"amount": 400, "name": "Dennis"}\n'
    # )
    buffer = (
        '{"amount": 100, "name": "Alice"}\n'
        '{"amount": 200, "name": "Bob"}\n'
        '{"amount": 300, "name": "Charlie"}\n'
        '{"amount": 400, "name": "Dennis"}\n'
    )

    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_json(
            f"s3://{bname}/{fname}",
            engine="cudf",
            orient="records",
            lines=True,
            storage_options=s3so,
        )

    expect = pd.read_json(buffer, lines=True)
    assert_eq(expect, got)


@pytest.mark.parametrize("use_python_file_object", [False, True])
@pytest.mark.parametrize("columns", [None, ["string1"]])
def test_read_orc(s3_base, s3so, datadir, use_python_file_object, columns):
    source_file = str(datadir / "orc" / "TestOrcFile.testSnappy.orc")
    fname = "test_orc_reader.orc"
    bname = "orc"
    expect = pa.orc.ORCFile(source_file).read().to_pandas()

    with open(source_file, "rb") as f:
        buffer = f.read()

    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        got = cudf.read_orc(
            f"s3://{bname}/{fname}",
            columns=columns,
            storage_options=s3so,
            use_python_file_object=use_python_file_object,
        )

    if columns:
        expect = expect[columns]
    assert_eq(expect, got)


@pytest.mark.parametrize("columns", [None, ["string1"]])
def test_read_orc_arrow_nativefile(s3_base, s3so, datadir, columns):
    source_file = str(datadir / "orc" / "TestOrcFile.testSnappy.orc")
    fname = "test_orc_reader.orc"
    bname = "orc"
    expect = pa.orc.ORCFile(source_file).read().to_pandas()

    with open(source_file, "rb") as f:
        buffer = f.read()

    with s3_context(s3_base=s3_base, bucket=bname, files={fname: buffer}):
        fs = pa_fs.S3FileSystem(
            endpoint_override=s3so["client_kwargs"]["endpoint_url"],
        )
        with fs.open_input_file(f"{bname}/{fname}") as fil:
            got = cudf.read_orc(fil, columns=columns)

    if columns:
        expect = expect[columns]
    assert_eq(expect, got)


def test_write_orc(s3_base, s3so, pdf):
    fname = "test_orc_writer.orc"
    bname = "orc"
    gdf = cudf.from_pandas(pdf)
    with s3_context(s3_base=s3_base, bucket=bname) as s3fs:
        gdf.to_orc(f"s3://{bname}/{fname}", storage_options=s3so)
        assert s3fs.exists(f"s3://{bname}/{fname}")

        with s3fs.open(f"s3://{bname}/{fname}") as f:
            got = pa.orc.ORCFile(f).read().to_pandas()

    assert_eq(pdf, got)
