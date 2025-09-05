# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

from cudf.utils.issue_reporting import (
    create_issue_report,
    format_issue_report,
    get_system_info,
    report_error,
)


def test_get_system_info():
    """Test that system info collection works."""
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


def test_format_issue_report():
    """Test that issue report formatting works correctly."""
    title = "Test Issue"
    description = "This is a test description"
    
    report = format_issue_report(title, description, include_system_info=False)
    
    assert "**Issue Title:** Test Issue" in report
    assert "**Description**" in report
    assert "This is a test description" in report
    assert "**Additional Context**" in report
    assert "cudf.utils.issue_reporting" in report


def test_format_issue_report_with_error():
    """Test formatting with error information."""
    title = "Test Error Issue"
    description = "Error occurred"
    error = ValueError("Test error message")
    
    report = format_issue_report(
        title, description, error=error, include_system_info=False
    )
    
    assert "**Error Information**" in report
    assert "ValueError" in report
    assert "Test error message" in report


def test_format_issue_report_with_example():
    """Test formatting with minimal example."""
    title = "Test Example Issue"
    description = "Issue with example"
    example = "import cudf\ndf = cudf.DataFrame({'a': [1, 2, 3]})"
    
    report = format_issue_report(
        title, description, minimal_example=example, include_system_info=False
    )
    
    assert "**Minimal Code Example**" in report
    assert "```python" in report
    assert "import cudf" in report


def test_create_issue_report():
    """Test the main create_issue_report function."""
    title = "Test Create Issue"
    description = "Testing create function"
    
    report = create_issue_report(title, description)
    
    assert title in report
    assert description in report
    # Check that system info is included (version may be "cuDF not available")
    assert "cuDF Version:" in report


def test_report_error():
    """Test the report_error function."""
    error = RuntimeError("Something went wrong")
    context = "During data processing"
    
    report = report_error(error, context=context)
    
    assert "[BUG]" in report
    assert "RuntimeError" in report
    assert "Something went wrong" in report
    assert "During data processing" in report


def test_report_error_with_custom_title():
    """Test report_error with custom title."""
    error = ValueError("Invalid value")
    title = "Custom Error Title"
    
    report = report_error(error, title=title)
    
    assert "Custom Error Title" in report
    assert "ValueError" in report


def test_accessible_from_cudf_utils():
    """Test that functions are accessible from cudf.utils."""
    # Test direct import from module works
    from cudf.utils.issue_reporting import create_issue_report, report_error
    
    # Test that they are callable
    assert callable(create_issue_report)
    assert callable(report_error)


def test_create_issue_report_file_output(tmp_path):
    """Test writing issue report to file."""
    title = "Test File Output"
    description = "Testing file output"
    output_file = tmp_path / "issue_report.txt"
    
    report = create_issue_report(
        title, description, output_file=str(output_file)
    )
    
    # Check file was created and contains the report
    assert output_file.exists()
    content = output_file.read_text()
    assert content == report
    assert title in content