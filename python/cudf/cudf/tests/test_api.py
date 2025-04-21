# Copyright (c) 2025, NVIDIA CORPORATION.
import cudf


def test_toplevel_imports_matches_all_modules():
    dir_objects = {obj for obj in dir(cudf) if not obj.startswith("_")}
    all_objects = set(cudf.__all__)
    extras = dir_objects - all_objects
    assert not extras, f"{extras} not included in cudf.__all__"
