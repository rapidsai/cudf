import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq

gcsfs = pytest.importorskip("gcsfs")


def test_gcs_public():
    fname = "anaconda-public-data/iris/iris.csv"

    expect = pd.read_csv("gcs://" + fname)
    got = cudf.read_csv("gcs://" + fname)
    assert_eq(expect, got)
