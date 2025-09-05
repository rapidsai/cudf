# Copyright (c) 2025, NVIDIA CORPORATION.

"""Utilities for generating issue reports to help with debugging cuDF problems."""

from __future__ import annotations

import platform
import subprocess
import sys
import traceback
from typing import Any, TextIO
import warnings


def get_system_info() -> dict[str, Any]:
    """
    Collect system information relevant for cuDF issue reporting.
    
    Returns
    -------
    dict
        Dictionary containing system information including OS, Python version,
        cuDF version, GPU information, and other relevant details.
    """
    # Try to get cuDF version, but don't fail if it's not available
    try:
        import cudf
        cudf_version = cudf.__version__
    except ImportError:
        cudf_version = "cuDF not available"
    
    info = {
        "cudf_version": cudf_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.architecture(),
    }
    
    # Try to get GPU information
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            info["gpu_info"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info["gpu_info"] = "GPU information not available"
    
    # Try to get CUDA version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            info["cuda_version"] = result.stdout.strip().split('\n')[-1]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info["cuda_version"] = "CUDA information not available"
    
    return info


def format_issue_report(
    title: str,
    description: str,
    error: Exception | None = None,
    include_system_info: bool = True,
    minimal_example: str | None = None
) -> str:
    """
    Format a comprehensive issue report for GitHub.
    
    Parameters
    ----------
    title : str
        Title of the issue
    description : str
        Description of the issue
    error : Exception, optional
        Exception that occurred, if any
    include_system_info : bool, default True
        Whether to include system information
    minimal_example : str, optional
        Minimal code example that reproduces the issue
        
    Returns
    -------
    str
        Formatted issue report ready to be posted to GitHub
    """
    report_lines = [
        f"**Issue Title:** {title}",
        "",
        "**Description**",
        description,
        ""
    ]
    
    if minimal_example:
        report_lines.extend([
            "**Minimal Code Example**",
            "```python",
            minimal_example,
            "```",
            ""
        ])
    
    if error:
        report_lines.extend([
            "**Error Information**",
            f"```",
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}",
            "",
            "Traceback:",
            traceback.format_exc(),
            "```",
            ""
        ])
    
    if include_system_info:
        system_info = get_system_info()
        report_lines.extend([
            "**Environment Information**",
            f"- cuDF Version: {system_info['cudf_version']}",
            f"- Python Version: {system_info['python_version']}",
            f"- Platform: {system_info['platform']}",
            f"- GPU Info: {system_info['gpu_info']}",
            f"- CUDA Version: {system_info['cuda_version']}",
            ""
        ])
    
    report_lines.extend([
        "**Additional Context**",
        "Please add any additional context about the problem here.",
        "",
        "---",
        "*This report was generated using cudf.utils.issue_reporting*"
    ])
    
    return "\n".join(report_lines)


def create_issue_report(
    title: str,
    description: str,
    error: Exception | None = None,
    minimal_example: str | None = None,
    output_file: str | TextIO | None = None
) -> str:
    """
    Create a comprehensive issue report for cuDF problems.
    
    This function generates a formatted issue report that can be copied
    and pasted into a GitHub issue. It includes system information,
    error details, and a structured format that follows cuDF's
    issue reporting guidelines.
    
    Parameters
    ----------
    title : str
        Title for the issue
    description : str
        Detailed description of the problem
    error : Exception, optional
        Exception that occurred, if any
    minimal_example : str, optional
        Minimal code example that reproduces the issue
    output_file : str or file-like object, optional
        If provided, write the report to this file. If string,
        treat as filename. If file-like object, write to it.
        
    Returns
    -------
    str
        The formatted issue report
        
    Examples
    --------
    >>> import cudf
    >>> report = cudf.utils.create_issue_report(
    ...     title="DataFrame.merge fails with large datasets",
    ...     description="When merging two large DataFrames, cuDF raises a memory error.",
    ...     minimal_example='''
    ...     import cudf
    ...     df1 = cudf.DataFrame({'a': range(1000000)})
    ...     df2 = cudf.DataFrame({'a': range(1000000)})
    ...     result = df1.merge(df2)  # Fails here
    ...     '''
    ... )
    >>> print(report)  # doctest: +SKIP
    """
    report = format_issue_report(
        title=title,
        description=description,
        error=error,
        minimal_example=minimal_example
    )
    
    if output_file:
        if isinstance(output_file, str):
            with open(output_file, 'w') as f:
                f.write(report)
        else:
            output_file.write(report)
    
    return report


def report_error(
    error: Exception,
    context: str | None = None,
    title: str | None = None,
    minimal_example: str | None = None
) -> str:
    """
    Generate an issue report for a specific error.
    
    Parameters
    ----------
    error : Exception
        The exception that occurred
    context : str, optional
        Additional context about when/how the error occurred
    title : str, optional
        Custom title for the issue. If not provided, will be
        generated from the error type and message.
    minimal_example : str, optional
        Minimal code example that reproduces the error
        
    Returns
    -------
    str
        Formatted issue report
        
    Examples
    --------
    >>> import cudf
    >>> try:
    ...     df = cudf.DataFrame({'a': [1, 2, 3]})
    ...     result = df.some_nonexistent_method()
    ... except AttributeError as e:
    ...     report = cudf.utils.report_error(
    ...         error=e,
    ...         context="Trying to call a method that doesn't exist"
    ...     )
    ...     print(report)  # doctest: +SKIP
    """
    if title is None:
        title = f"[BUG] {type(error).__name__}: {str(error)[:50]}..."
    
    description = f"An error occurred in cuDF: {str(error)}"
    if context:
        description += f"\n\nContext: {context}"
    
    return create_issue_report(
        title=title,
        description=description,
        error=error,
        minimal_example=minimal_example
    )


if __name__ == "__main__":
    # Simple test when run directly
    print("Testing issue reporting functionality...")
    
    # Test system info
    info = get_system_info()
    print(f"✓ System info collected: {len(info)} items")
    
    # Test basic report
    report = format_issue_report(
        "Test Issue", 
        "This is a test", 
        include_system_info=False
    )
    print(f"✓ Basic report generated: {len(report)} characters")
    
    # Test with error
    test_error = ValueError("Test error")
    error_report = report_error(test_error, "Testing error reporting")
    print(f"✓ Error report generated: {len(error_report)} characters")
    
    print("All tests passed! Issue reporting utility is working.")