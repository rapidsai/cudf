# Issue Reporting Utility

This document describes the new issue reporting utility added to cuDF that helps users generate comprehensive issue reports for GitHub.

## Overview

The `cudf.utils.issue_reporting` module provides functions to programmatically generate detailed issue reports that include:

- System information (OS, Python version, GPU details, CUDA version)
- Error details and stack traces
- Minimal code examples 
- Structured formatting for GitHub issues

## Functions

### `create_issue_report(title, description, error=None, minimal_example=None, output_file=None)`

Creates a comprehensive issue report for cuDF problems.

**Parameters:**
- `title` (str): Title for the issue
- `description` (str): Detailed description of the problem  
- `error` (Exception, optional): Exception that occurred, if any
- `minimal_example` (str, optional): Minimal code example that reproduces the issue
- `output_file` (str or file-like, optional): If provided, write the report to this file

**Returns:** str - The formatted issue report

### `report_error(error, context=None, title=None, minimal_example=None)`

Generate an issue report for a specific error.

**Parameters:**
- `error` (Exception): The exception that occurred
- `context` (str, optional): Additional context about when/how the error occurred
- `title` (str, optional): Custom title for the issue
- `minimal_example` (str, optional): Minimal code example that reproduces the error

**Returns:** str - Formatted issue report

### `get_system_info()`

Collect system information relevant for cuDF issue reporting.

**Returns:** dict - Dictionary containing system information

## Usage Examples

### Basic Issue Report

```python
import cudf

report = cudf.utils.create_issue_report(
    title="DataFrame.merge() fails with memory error",
    description="When merging large DataFrames, cuDF runs out of memory",
    minimal_example='''
    import cudf
    df1 = cudf.DataFrame({'id': range(1000000)})
    df2 = cudf.DataFrame({'id': range(1000000)})
    result = df1.merge(df2)  # Fails here
    '''
)

print(report)
```

### Error Report from Exception

```python
try:
    # Some cuDF operation that fails
    df = cudf.DataFrame({'a': [1, 2, 3]})
    result = df.some_invalid_operation()
except Exception as e:
    report = cudf.utils.report_error(
        error=e,
        context="Trying to call an invalid operation on DataFrame"
    )
    print(report)
```

### Save Report to File

```python
cudf.utils.create_issue_report(
    title="Performance Issue",
    description="Operation is slower than expected",
    output_file="issue_report.txt"
)
```

## Generated Report Format

The generated reports follow this structure:

```
**Issue Title:** [Title]

**Description**
[Detailed description]

**Minimal Code Example**
```python
[Code example]
```

**Error Information** (if error provided)
```
Error Type: ExceptionType
Error Message: [message]

Traceback:
[Full traceback]
```

**Environment Information**
- cuDF Version: [version]
- Python Version: [version]
- Platform: [platform info]
- GPU Info: [GPU details]
- CUDA Version: [CUDA version]

**Additional Context**
Please add any additional context about the problem here.

---
*This report was generated using cudf.utils.issue_reporting*
```

## Integration with GitHub

The generated reports are formatted to work well with GitHub's issue templates and can be:

1. Copied and pasted directly into a new GitHub issue
2. Saved to a file for later submission
3. Used as a starting point for bug reports

The format follows cuDF's contributing guidelines and includes all the information typically requested in bug reports.