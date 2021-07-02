# Copyright (c) 2020, NVIDIA CORPORATION.

import io
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pytest

import cudf
from cudf.testing._utils import assert_eq

gcsfs = pytest.importorskip("gcsfs")

TEST_PROJECT = "cudf-gcs-test-project"
TEST_BUCKET = "cudf-gcs-test-bucket"


@pytest.fixture
def pdf(scope="module"):
    df = pd.DataFrame()
    df["Integer"] = np.array([2345, 11987, 9027, 9027])
    df["Float"] = np.array([9.001, 8.343, 6, 2.781])
    df["Integer2"] = np.array([2345, 106, 2088, 789277])
    df["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    df["Boolean"] = np.array([True, False, True, False])
    return df


def test_read_csv(pdf, monkeypatch):
    # Write to buffer
    fpath = TEST_BUCKET + "test_csv_reader.csv"
    buffer = pdf.to_csv(index=False)

    def mock_open(*args):
        return io.BytesIO(buffer.encode())

    monkeypatch.setattr(gcsfs.core.GCSFileSystem, "open", mock_open)
    got = cudf.read_csv("gcs://{}".format(fpath))

    assert_eq(pdf, got)


def test_write_orc(pdf, monkeypatch, tmpdir):
    gcs_fname = TEST_BUCKET + "test_orc_writer.orc"
    local_filepath = os.path.join(tmpdir, "test_orc.orc")
    gdf = cudf.from_pandas(pdf)

    def mock_open(*args, **kwargs):
        return open(local_filepath, "wb")

    monkeypatch.setattr(gcsfs.core.GCSFileSystem, "open", mock_open)
    gdf.to_orc("gcs://{}".format(gcs_fname))

    got = pa.orc.ORCFile(local_filepath).read().to_pandas()
    assert_eq(pdf, got)
