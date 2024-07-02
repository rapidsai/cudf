# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import io
import os

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq

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


def test_read_csv(pdf, monkeypatch, tmpdir):
    # Write to buffer
    fpath = TEST_BUCKET + "test_csv_reader.csv"
    buffer = pdf.to_csv(index=False)

    def mock_open(*args, **kwargs):
        return io.BytesIO(buffer.encode())

    def mock_size(*args):
        return len(buffer.encode())

    monkeypatch.setattr(gcsfs.core.GCSFileSystem, "open", mock_open)
    monkeypatch.setattr(gcsfs.core.GCSFileSystem, "size", mock_size)

    # Test read from explicit path.
    # Since we are monkey-patching, we cannot use
    # use_python_file_object=True, because the pyarrow
    # `open_input_file` command will fail (since it doesn't
    # use the monkey-patched `open` definition)
    with pytest.warns(FutureWarning):
        got = cudf.read_csv(f"gcs://{fpath}", use_python_file_object=False)
    assert_eq(pdf, got)

    # AbstractBufferedFile -> PythonFile conversion
    # will work fine with the monkey-patched FS if we
    # pass in an fsspec file object
    fs = gcsfs.core.GCSFileSystem()
    with fs.open(f"gcs://{fpath}") as f:
        got = cudf.read_csv(f)
    assert_eq(pdf, got)


def test_write_orc(pdf, monkeypatch, tmpdir):
    gcs_fname = TEST_BUCKET + "test_orc_writer.orc"
    local_filepath = os.path.join(tmpdir, "test_orc.orc")
    gdf = cudf.from_pandas(pdf)

    def mock_open(*args, **kwargs):
        return open(local_filepath, "wb")

    monkeypatch.setattr(gcsfs.core.GCSFileSystem, "open", mock_open)
    gdf.to_orc(f"gcs://{gcs_fname}")

    got = pd.read_orc(local_filepath)
    assert_eq(pdf, got)
