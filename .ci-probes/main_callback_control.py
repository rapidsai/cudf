"""Temporary diagnostic control; never used to generate GPU run lists."""

import os
import runpy

import pytest
import polars
import cudf_polars
from cudf_polars.testing.fallback_report import COLLECTION


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    namespace = runpy.run_path(os.environ["CUDF_POLARS_MAIN_CALLBACK_SOURCE"])
    cudf_polars.execute_with_cudf = namespace["execute_with_cudf"]
    polars.GPUEngine._post_opt_callback = config._cudf_polars_post_opt_callback
    print("Callback control: upstream main bfac55d1aacf560770ca422d32fe6cb0f1cbeff0; attribution wrapping disabled")


def pytest_collection_finish(session):
    # Control observations cannot be admitted to maintained GPU selections.
    session.config.stash[COLLECTION]["selection_mode"] = "control"
