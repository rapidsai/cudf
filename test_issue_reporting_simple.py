#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.

"""Simple test script for issue reporting functionality."""

import sys
import tempfile
import os
import importlib.util

# Load the module directly without going through cudf package
spec = importlib.util.spec_from_file_location(
    'issue_reporting', 
    '/home/runner/work/cudf/cudf/python/cudf/cudf/utils/issue_reporting.py'
)
issue_reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(issue_reporting)

# Import the functions
create_issue_report = issue_reporting.create_issue_report
format_issue_report = issue_reporting.format_issue_report
get_system_info = issue_reporting.get_system_info
report_error = issue_reporting.report_error


def test_get_system_info():
    """Test that system info collection works."""
    print("Testing get_system_info...")
    info = get_system_info()
    
    # Check that required fields are present
    assert "cudf_version" in info
    assert "python_version" in info
    assert "platform" in info
    assert "os" in info
    assert "architecture" in info
    assert "gpu_info" in info
    assert "cuda_version" in info
    
    # Check that cudf version is set (may be "cuDF not available")
    assert info["cudf_version"] is not None
    print("✓ get_system_info test passed")


def test_format_issue_report():
    """Test that issue report formatting works correctly."""
    print("Testing format_issue_report...")
    title = "Test Issue"
    description = "This is a test description"
    
    report = format_issue_report(title, description, include_system_info=False)
    
    assert "**Issue Title:** Test Issue" in report
    assert "**Description**" in report
    assert "This is a test description" in report
    assert "**Additional Context**" in report
    assert "cudf.utils.issue_reporting" in report
    print("✓ format_issue_report test passed")


def test_format_issue_report_with_error():
    """Test formatting with error information."""
    print("Testing format_issue_report with error...")
    title = "Test Error Issue"
    description = "Error occurred"
    error = ValueError("Test error message")
    
    report = format_issue_report(
        title, description, error=error, include_system_info=False
    )
    
    assert "**Error Information**" in report
    assert "ValueError" in report
    assert "Test error message" in report
    print("✓ format_issue_report with error test passed")


def test_format_issue_report_with_example():
    """Test formatting with minimal example."""
    print("Testing format_issue_report with example...")
    title = "Test Example Issue"
    description = "Issue with example"
    example = "import cudf\ndf = cudf.DataFrame({'a': [1, 2, 3]})"
    
    report = format_issue_report(
        title, description, minimal_example=example, include_system_info=False
    )
    
    assert "**Minimal Code Example**" in report
    assert "```python" in report
    assert "import cudf" in report
    print("✓ format_issue_report with example test passed")


def test_create_issue_report():
    """Test the main create_issue_report function."""
    print("Testing create_issue_report...")
    title = "Test Create Issue"
    description = "Testing create function"
    
    report = create_issue_report(title, description)
    
    assert title in report
    assert description in report
    # Check that system info is included (version may be "cuDF not available")
    assert "cuDF Version:" in report
    print("✓ create_issue_report test passed")


def test_report_error():
    """Test the report_error function."""
    print("Testing report_error...")
    error = RuntimeError("Something went wrong")
    context = "During data processing"
    
    report = report_error(error, context=context)
    
    assert "[BUG]" in report
    assert "RuntimeError" in report
    assert "Something went wrong" in report
    assert "During data processing" in report
    print("✓ report_error test passed")


def test_report_error_with_custom_title():
    """Test report_error with custom title."""
    print("Testing report_error with custom title...")
    error = ValueError("Invalid value")
    title = "Custom Error Title"
    
    report = report_error(error, title=title)
    
    assert "Custom Error Title" in report
    assert "ValueError" in report
    print("✓ report_error with custom title test passed")


def test_create_issue_report_file_output():
    """Test writing issue report to file."""
    print("Testing file output...")
    title = "Test File Output"
    description = "Testing file output"
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        output_file = f.name
    
    try:
        report = create_issue_report(
            title, description, output_file=output_file
        )
        
        # Check file was created and contains the report
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            content = f.read()
        assert content == report
        assert title in content
        print("✓ file output test passed")
    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


def main():
    """Run all tests."""
    print("Running issue reporting tests...\n")
    
    test_get_system_info()
    test_format_issue_report()
    test_format_issue_report_with_error()
    test_format_issue_report_with_example()
    test_create_issue_report()
    test_report_error()
    test_report_error_with_custom_title()
    test_create_issue_report_file_output()
    
    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    main()