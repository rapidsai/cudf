# Copyright (c) 2020, NVIDIA CORPORATION.

from contextlib import contextmanager

import cudf._lib.nvtx._lib as libnvtx


@contextmanager
def annotate(message=None, color=None, domain=None):
    """
    Annotate a function or a code range.

    Parameters
    ----------
    message : str
        A message associated with the annotated code range.
    color : str, color
        A color associated with the annotated code range.
        Supports `matplotlib` colors if it is available.
    domain : str
        Name of a domain under which the code range is scoped.
        The default domain is called "NVTX".

    Examples
    --------
    >>> import nvtx
    >>> import time

    Using a decorator:

    >>> @nvtx.annotate("my_func", color="red", domain="cudf")
    ... def func():
    ...     time.sleep(0.1)

    Using a context manager:

    >>> with nvtx.annotate("my_code_range", color="blue"):
    ...    time.sleep(10)
    ...
    """
    push_range(message, color, domain)
    try:
        yield
    finally:
        pop_range(domain)


def push_range(message=None, color=None, domain=None):
    """
    Mark the beginning of a code range.

    Parameters
    ----------
    message : str
        A message associated with the annotated code range.
    color : str, color
        A color associated with the annotated code range.
        Supports
    domain : str
        Name of a domain under which the code range is scoped.
        The default domain is called "NVTX".

    Examples
    --------
    >>> import nvtx
    >>> nvtx.push_range("my_code_range", domain="cudf")
    >>> time.sleep(1)
    >>> nvtx.pop_range(domain="cudf")
    """
    libnvtx.push_range(message, color, domain)


def pop_range(domain=None):
    """
    Mark the end of a code range that was started with `push_range`.

    Parameters
    ----------
    domain : str
        The domain under which the code range is scoped. The default
        domain is "NVTX".
    """
    libnvtx.pop_range(domain)
