# Copyright (c) 2020, NVIDIA CORPORATION.

def __getattr__(name):
    """Lazy import for utils functions."""
    if name == "create_issue_report":
        from cudf.utils.issue_reporting import create_issue_report
        return create_issue_report
    elif name == "report_error":
        from cudf.utils.issue_reporting import report_error
        return report_error
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
