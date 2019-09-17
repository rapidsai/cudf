import io

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq

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


def test_csv(pdf, monkeypatch):
    # Write to buffer
    fpath = TEST_BUCKET + "file.csv"
    buffer = pdf.to_csv(index=False)

    def mock_open(*args):
        return io.BytesIO(buffer.encode())

    monkeypatch.setattr(gcsfs.core.GCSFileSystem, "open", mock_open)
    got = cudf.read_csv("gcs://{}".format(fpath))

    assert_eq(pdf, got)
