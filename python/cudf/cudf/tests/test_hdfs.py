import os
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.tests.utils import assert_eq

if not os.environ.get("RUN_HDFS_TESTS"):
    pytestmark = pytest.mark.skip("Env not configured to run HDFS tests")


basedir = "/tmp/test-hdfs"


@pytest.fixture
def hdfs():
    fs = pa.hdfs.connect()
    try:
        if not fs.exists(basedir):
            fs.mkdir(basedir)
    except pa.lib.ArrowIOError:
        pytest.skip("hdfs config probably incorrect")

    return fs


@pytest.fixture
def pdf():
    df = pd.DataFrame()
    df["Integer"] = np.array([2345, 11987, 9027, 9027])
    df["Date"] = np.array(
        ["18/04/1995", "14/07/1994", "07/06/2006", "16/09/2005"]
    )
    df["Float"] = np.array([9.001, 8.343, 6, 2.781])
    df["Integer2"] = np.array([2345, 106, 2088, 789277])
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


def test_csv(tmpdir, pdf, hdfs):
    fname = tmpdir.mkdir("csv").join("file1.csv")
    # Write to local file system
    pdf.to_csv(fname)
    # Read from local file system as buffer
    with open(fname, mode="rb") as f:
        buffer = BytesIO(f.read())
    # Write to hdfs
    hdfs.upload(basedir + "file.csv", buffer)

    gdf = cudf.read_csv("hdfs://" + basedir + "file.csv")

    # Read pandas from byte buffer
    with hdfs.open(basedir + "file.csv") as f:
        pdf = pd.read_csv("file.csv")

    assert_eq(pdf, gdf)
