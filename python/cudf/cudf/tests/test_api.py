# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cudf


def test_toplevel_imports_matches_all_modules():
    dir_objects = {obj for obj in dir(cudf) if not obj.startswith("_")}
    all_objects = set(cudf.__all__)
    extras = dir_objects - all_objects
    assert not extras, f"{extras} not included in cudf.__all__"


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(cudf.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(cudf.__version__, str)
    assert len(cudf.__version__) > 0
