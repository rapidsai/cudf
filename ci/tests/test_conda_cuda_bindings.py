# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render actual CUDA selectors before and after candidate YAML serialization."""

import json
import os
import shutil
import subprocess  # nosec B404 -- fixed local renderer invocation.
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("recipe_name", ["cudf-polars", "cudf", "pylibcudf"])
@pytest.mark.parametrize("platform", ["linux-64", "linux-aarch64"])
@pytest.mark.parametrize("cuda", ["12.9.2", "13.3.0"])
@pytest.mark.parametrize("serialized", [False, True])
def test_rendered_bindings_match_cuda_family(
    recipe_name, platform, cuda, serialized
):
    recipe_path = ROOT / f"conda/recipes/{recipe_name}/recipe.yaml"
    executable = os.environ.get("RATTLER_BUILD") or shutil.which(
        "rattler-build"
    )
    if executable is None:
        pytest.skip("rattler-build is required for the actual recipe renderer")
    environment = os.environ.copy()
    environment.update(
        RAPIDS_CUDA_VERSION=cuda,
        RAPIDS_PACKAGE_VERSION=(ROOT / "VERSION").read_text().strip(),
        RAPIDS_PY_VERSION="3.12",
        RAPIDS_DATETIME_STRING="261014000000",
        RAPIDS_CONDA_ARCH=platform,
        CMAKE_C_COMPILER_LAUNCHER="",
        CMAKE_CXX_COMPILER_LAUNCHER="",
        CMAKE_CUDA_COMPILER_LAUNCHER="",
        CMAKE_GENERATOR="Ninja",
    )
    # The real candidate patcher copies beside the original recipe and writes
    # safe_dump(safe_load(...)). That removes quotes around context expressions,
    # making Rattler interpret their numeric results as integers, not strings.
    with tempfile.TemporaryDirectory(
        prefix=".cuda-render-test-", dir=recipe_path.parent.parent
    ) as directory:
        copied = Path(directory)
        shutil.copytree(recipe_path.parent, copied, dirs_exist_ok=True)
        recipe = copied / "recipe.yaml"
        if serialized:
            recipe.write_text(
                yaml.safe_dump(
                    yaml.safe_load(recipe.read_text()), sort_keys=False
                )
            )
        result = subprocess.run(  # nosec B603 -- fixed renderer argv, no shell.
            [
                executable,
                "build",
                "--recipe",
                str(recipe),
                "--experimental",
                "--render-only",
                "--output-dir",
                str(copied / "output"),
                "--target-platform",
                platform,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    assert result.returncode == 0, result.stderr  # nosec B101 -- pytest assertion.
    rendered = json.loads(result.stdout)
    expected = (
        "cuda-bindings >=12.9.3,<13.0"
        if cuda.startswith("12.")
        else "cuda-bindings >=13.0.2,<14.0"
    )
    assert rendered  # nosec B101 -- pytest assertion.
    for output in rendered:
        for section in ("run",):
            bindings = [
                requirement
                for requirement in output["recipe"]["requirements"][section]
                if isinstance(requirement, str)
                and requirement.startswith("cuda-bindings ")
            ]
            assert bindings == [expected]  # nosec B101 -- pytest assertion.
