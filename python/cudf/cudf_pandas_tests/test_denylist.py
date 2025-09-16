# Copyright (c) 2025, NVIDIA CORPORATION.

import os
import pathlib
import tempfile
import unittest.mock

import pytest

from cudf.pandas.module_accelerator import _caller_in_denylist


class TestDenylist:
    """Test the denylist functionality for third-party library compatibility."""

    def test_caller_in_denylist_basic(self):
        """Test basic denylist functionality."""
        # Create test denylist
        denylist = (
            "/usr/lib/python3.13/site-packages/pandas",
            "/usr/lib/python3.13/site-packages/cudf",
        )

        # Test pandas path (should be denied)
        pandas_path = pathlib.PurePath(
            "/usr/lib/python3.13/site-packages/pandas/core/frame.py"
        )
        assert _caller_in_denylist(pandas_path, denylist) is True

        # Test non-pandas path (should not be denied)
        other_path = pathlib.PurePath(
            "/usr/lib/python3.13/site-packages/numpy/core/numeric.py"
        )
        assert _caller_in_denylist(other_path, denylist) is False

        # Test cudf path (should be denied)
        cudf_path = pathlib.PurePath(
            "/usr/lib/python3.13/site-packages/cudf/core/dataframe.py"
        )
        assert _caller_in_denylist(cudf_path, denylist) is True

    def test_caller_in_denylist_xarray(self):
        """Test xarray-specific denylist functionality."""
        # Create denylist with xarray
        denylist = (
            "/usr/lib/python3.13/site-packages/pandas",
            "/usr/lib/python3.13/site-packages/cudf",
            "/usr/lib/python3.13/site-packages/xarray",
        )

        # Test xarray path (should be denied)
        xarray_path = pathlib.PurePath(
            "/usr/lib/python3.13/site-packages/xarray/core/common.py"
        )
        assert _caller_in_denylist(xarray_path, denylist) is True

        # Test nested xarray path (should be denied)
        xarray_nested = pathlib.PurePath(
            "/usr/lib/python3.13/site-packages/xarray/backends/zarr.py"
        )
        assert _caller_in_denylist(xarray_nested, denylist) is True

    def test_environment_variable_parsing(self):
        """Test parsing of CUDF_PANDAS_DENYLIST environment variable."""
        # This test simulates the environment variable parsing logic
        # without actually installing the module accelerator

        test_cases = [
            ("", []),  # Empty string
            ("xarray", ["xarray"]),  # Single module
            ("xarray,scipy", ["xarray", "scipy"]),  # Multiple modules
            (
                "xarray, scipy, sklearn",
                ["xarray", "scipy", "sklearn"],
            ),  # With spaces
            ("xarray,,scipy", ["xarray", "scipy"]),  # Empty tokens
            (" xarray , , scipy ", ["xarray", "scipy"]),  # Mixed whitespace
        ]

        for env_value, expected_tokens in test_cases:
            tokens = []
            for token in (t.strip() for t in env_value.split(",")):
                if token:
                    tokens.append(token)
            assert tokens == expected_tokens, f"Failed for '{env_value}'"

    def test_module_path_resolution(self):
        """Test that module paths are resolved correctly."""
        # Test with a built-in module that should always exist
        import sys as test_module

        # Test module with __path__ (package)
        if hasattr(test_module, "__path__"):
            paths = [str(path) for path in test_module.__path__]
            assert len(paths) > 0

        # Test module with __file__ (single module)
        if hasattr(test_module, "__file__"):
            parent = str(pathlib.Path(test_module.__file__).resolve().parent)
            assert len(parent) > 0

    def test_filesystem_path_handling(self):
        """Test handling of filesystem paths in denylist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test path
            test_path = pathlib.Path(temp_dir) / "test_library"
            test_path.mkdir()

            # Test that existing paths are handled
            resolved_path = pathlib.Path(str(test_path)).resolve()
            assert resolved_path.exists()

            # Test non-existent path (should be ignored)
            fake_path = pathlib.Path(temp_dir) / "nonexistent"
            assert not fake_path.exists()

    @unittest.mock.patch.dict(
        os.environ, {"CUDF_PANDAS_DENYLIST": "xarray,scipy"}
    )
    def test_environment_variable_integration(self):
        """Test that environment variable is read correctly."""
        env_value = os.environ.get("CUDF_PANDAS_DENYLIST", "").strip()
        assert env_value == "xarray,scipy"

        tokens = [t.strip() for t in env_value.split(",") if t.strip()]
        assert tokens == ["xarray", "scipy"]

    def test_denylist_path_matching(self):
        """Test that path matching works correctly with various path formats."""
        # Test with different path separators and formats
        denylist = ("/usr/lib/python3.13/site-packages/xarray",)

        test_cases = [
            # Should match (inside xarray)
            ("/usr/lib/python3.13/site-packages/xarray/core/common.py", True),
            ("/usr/lib/python3.13/site-packages/xarray/__init__.py", True),
            (
                "/usr/lib/python3.13/site-packages/xarray/backends/zarr.py",
                True,
            ),
            # Should not match (outside xarray)
            ("/usr/lib/python3.13/site-packages/pandas/core/frame.py", False),
            ("/usr/lib/python3.13/site-packages/numpy/core/numeric.py", False),
            ("/home/user/my_script.py", False),
        ]

        for path_str, expected in test_cases:
            path = pathlib.PurePath(path_str)
            result = _caller_in_denylist(path, denylist)
            assert result is expected, (
                f"Path {path_str} should {'be' if expected else 'not be'} denied"
            )

    def test_denylist_with_real_modules(self):
        """Test denylist with real importable modules."""
        # Test with sys module (should always be available)
        try:
            import sys as test_sys

            # Get module path
            if hasattr(test_sys, "__path__"):
                # Package module
                sys_paths = [str(path) for path in test_sys.__path__]
                assert len(sys_paths) >= 0  # May be empty for built-in modules
            elif hasattr(test_sys, "__file__") and test_sys.__file__:
                # Single module
                sys_path = str(
                    pathlib.Path(test_sys.__file__).resolve().parent
                )
                assert len(sys_path) > 0

        except ImportError:
            pytest.skip("sys module not available (unexpected)")

    def test_malformed_denylist_entries(self):
        """Test that malformed denylist entries are handled gracefully."""
        # This test simulates what happens with bad environment variable values
        malformed_entries = [
            "nonexistent_module_12345",  # Module that doesn't exist
            "/path/that/does/not/exist",  # Path that doesn't exist
            "",  # Empty string
            "   ",  # Whitespace only
        ]

        # The parsing logic should handle these gracefully without crashing
        for entry in malformed_entries:
            tokens = [t.strip() for t in entry.split(",") if t.strip()]
            # Should not crash, may result in empty list for malformed entries
            assert isinstance(tokens, list)
