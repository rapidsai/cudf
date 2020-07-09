import os
from io import BytesIO

import fastavro as fa
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import orc as orc

import cudf
from cudf.tests.utils import assert_eq

if not os.environ.get("RUN_HDFS_TESTS"):
    pytestmark = pytest.mark.skip("Env not configured to run HDFS tests")


basedir = "/tmp/test-hdfs"
host = "localhost"  # hadoop hostname
port = 8020  # hadoop rpc port


@pytest.fixture
def hdfs(scope="module"):
    # Default Rpc port can be 8020/9000 depending on the hdfs config
    fs = pa.hdfs.connect(host=host, port=port)
    try:
        if not fs.exists(basedir):
            fs.mkdir(basedir)
    except pa.lib.ArrowIOError:
        pytest.skip("hdfs config probably incorrect")

    return fs


@pytest.fixture
def pdf(scope="module"):
    df = pd.DataFrame()
    df["Integer"] = np.array([2345, 11987, 9027, 9027])
    df["Float"] = np.array([9.001, 8.343, 6, 2.781])
    df["Integer2"] = np.array([2345, 106, 2088, 789277], dtype="uint64")
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


@pytest.mark.parametrize("test_url", [False, True])
def test_read_csv(tmpdir, pdf, hdfs, test_url):
    fname = tmpdir.mkdir("csv").join("file.csv")
    # Write to local file system
    pdf.to_csv(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.csv", buffer)

    if test_url:
        hd_fpath = "hdfs://{}:{}{}/file.csv".format(host, port, basedir)
    else:
        hd_fpath = "hdfs://{}/file.csv".format(basedir)

    got = cudf.read_csv(hd_fpath)

    # Read pandas from byte buffer
    with hdfs.open(basedir + "/file.csv") as f:
        expect = pd.read_csv(f)

    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_write_csv(pdf, test_url):
    gdf = cudf.from_pandas(pdf)
    if test_url:
        hd_fpath = "hdfs://{}:{}{}/file.csv".format(host, port, basedir)
    else:
        hd_fpath = "hdfs://{}/file.csv".format(basedir)

    with pytest.raises(
        RuntimeError, match="write_csv: file could not be opened"
    ):
        gdf.to_csv(hd_fpath)


@pytest.mark.parametrize("test_url", [False, True])
def test_parquet(tmpdir, pdf, hdfs, test_url):
    fname = tmpdir.mkdir("parquet").join("file.parq")
    # Write to local file system
    pdf.to_parquet(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.parq", buffer)

    if test_url:
        hd_fpath = "hdfs://{}:{}{}/file.parq".format(host, port, basedir)
    else:
        hd_fpath = "hdfs://{}/file.parq".format(basedir)

    got = cudf.read_parquet(hd_fpath)

    # Read pandas from byte buffer
    with hdfs.open(basedir + "/file.parq") as f:
        expect = pd.read_parquet(f)

    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_json(tmpdir, pdf, hdfs, test_url):
    fname = tmpdir.mkdir("json").join("file.json")
    # Write to local file system
    # Sorting by col_name now as pandas sorts by col name while reading json

    pdf.sort_index(axis=1).to_json(fname, orient="records", lines=True)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.json", buffer)

    if test_url:
        hd_fpath = "hdfs://{}:{}{}/file.json".format(host, port, basedir)
    else:
        hd_fpath = "hdfs://{}/file.json".format(basedir)

    got = cudf.read_json(hd_fpath, engine="cudf", orient="records", lines=True)

    # Read pandas from byte buffer
    with hdfs.open(basedir + "/file.json") as f:
        expect = pd.read_json(f, lines=True)

    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_orc(datadir, hdfs, test_url):
    fname = datadir / "orc" / "TestOrcFile.testSnappy.orc"
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.orc", buffer)

    if test_url:
        hd_fpath = "hdfs://{}:{}{}/file.orc".format(host, port, basedir)
    else:
        hd_fpath = "hdfs://{}/file.orc".format(basedir)

    got = cudf.read_orc(hd_fpath)
    expect = orc.ORCFile(buffer).read().to_pandas()
    assert_eq(expect, got)


@pytest.mark.parametrize("test_url", [False, True])
def test_avro(datadir, hdfs, test_url):
    fname = datadir / "avro" / "example.avro"
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.avro", buffer)

    if test_url:
        hd_fpath = "hdfs://{}:{}{}/file.avro".format(host, port, basedir)
    else:
        hd_fpath = "hdfs://{}/file.avro".format(basedir)

    got = cudf.read_avro(hd_fpath)
    with open(fname, mode="rb") as f:
        expect = pd.DataFrame.from_records(fa.reader(f))

    for col in expect.columns:
        expect[col] = expect[col].astype(got[col].dtype)
    assert_eq(expect, got)


def test_storage_options(tmpdir, pdf, hdfs):
    fname = tmpdir.mkdir("csv").join("file.csv")
    # Write to local file system
    pdf.to_csv(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.csv", buffer)

    hd_fpath = "hdfs://{}/file.csv".format(basedir)

    storage_options = {"host": host, "port": port}

    got = cudf.read_csv(hd_fpath, storage_options=storage_options)

    # Read pandas from byte buffer
    with hdfs.open(basedir + "/file.csv") as f:
        expect = pd.read_csv(f)

    assert_eq(expect, got)


def test_storage_options_error(tmpdir, pdf, hdfs):
    fname = tmpdir.mkdir("csv").join("file.csv")
    # Write to local file system
    pdf.to_csv(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "/file.csv", buffer)

    hd_fpath = "hdfs://{}:{}{}/file.avro".format(host, port, basedir)

    storage_options = {"host": host, "port": port}

    with pytest.raises(KeyError):
        cudf.read_csv(hd_fpath, storage_options=storage_options)
