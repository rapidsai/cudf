# Copyright (c) 2020, NVIDIA CORPORATION.
from contextlib import ContextDecorator

from cudf._lib.nvtx._lib import (
    Domain,
    EventAttributes,
    pop_range as libnvtx_pop_range,
    push_range as libnvtx_push_range,
)


class annotate(ContextDecorator):
    """
    Annotate code ranges using a context manager or a decorator.
    """

    def __init__(self, message=None, color="blue", domain=None):
        """
        Annotate a function or a code range.

        Parameters
        ----------
        message : str, optional
            A message associated with the annotated code range.
            When used as a decorator, the default value of message
            is the name of the function being decorated.
            When used as a context manager, the default value
            is the empty string.
        color : str or color, optional
            A color associated with the annotated code range.
            Supports `matplotlib` colors if it is available.
        domain : str, optional
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
        self._message = message
        self._color = color
        self._domain_name = domain
        self.attributes = EventAttributes(message, color)
        self.domain = Domain(domain)

    def __reduce__(self):
        return self.__class__, (self._message, self._color, self._domain_name)

    def __enter__(self):
        libnvtx_push_range(self.attributes, self.domain.handle)
        return self

    def __exit__(self, *exc):
        libnvtx_pop_range(self.domain.handle)
        return False

    def __call__(self, func):
        if not self.attributes.message:
            self.attributes.message = func.__name__
        return super().__call__(func)


class _annotate_nop:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        return False

    def __call__(self, func):
        return func


def push_range(message=None, color="blue", domain=None):
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
    libnvtx_push_range(EventAttributes(message, color), Domain(domain).handle)


def pop_range(domain=None):
    """
    Mark the end of a code range that was started with `push_range`.

    Parameters
    ----------
    domain : str
        The domain under which the code range is scoped. The default
        domain is "NVTX".
    """
    libnvtx_pop_range(Domain(domain).handle)
