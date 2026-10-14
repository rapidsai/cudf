# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep candidate producers out of inherited publication credentials and jobs."""

from pathlib import Path

import yaml

REPOSITORY = Path(__file__).parents[1]


def _workflow_jobs(path: str) -> dict:
    workflow = yaml.safe_load((REPOSITORY / path).read_text())
    return workflow["jobs"]


def test_candidate_build_jobs_do_not_inherit_repository_secrets():
    jobs = _workflow_jobs(".github/workflows/build.yaml")
    for name in (
        "cpp-build",
        "python-build",
        "python-build-noarch",
        "wheel-build",
        "wheel-python-build-noarch",
        "java-build",
    ):
        assert jobs[name].get("secrets") != "inherit"


def test_candidate_mode_skips_publication_jobs():
    jobs = _workflow_jobs(".github/workflows/build.yaml")
    publisher_jobs = (
        "upload-conda",
        "publish-api-docs",
        "wheel-publish-libcudf",
        "wheel-publish-libcudf-streaming",
        "wheel-publish-cudf-streaming",
        "wheel-publish-pylibcudf",
        "wheel-publish-cudf",
        "wheel-publish-dask-cudf",
        "wheel-publish-cudf-polars",
        "java-gather",
        "java-publish",
        "devcontainers",
    )
    for name in publisher_jobs:
        assert "release-candidate" in jobs[name]["if"]
