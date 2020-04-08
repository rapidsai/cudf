from contextlib import contextmanager

import cudf._lib.nvtx._lib as libnvtx
from cudf._lib.nvtx.colors import color_to_hex
from cudf._lib.nvtx.utils.cached import CachedInstanceMeta


class Domain(metaclass=CachedInstanceMeta):
    def __init__(self, name=None):
        self.name = name
        self.handle = libnvtx.DomainHandle(name)


class Range:
    """
    Specifies an NVTX code range via start/end or push/pop.

    Parameters
    ----------
    message : str
        Message associated with the code range.
    color : str
        Color associated with the code range.
    domain : str
        The name of the domain under which the code range is scoped.

    Examples
    --------
    >>> import nvtx
    >>> import time

    Using start/end API to define a code range:

    >>> rng = nvtx.Range("my_code_range")
    >>> rng.start()
    >>> print("code range starts")
    >>> time.sleep(10)
    >>> print("code range ends")
    >>> rng.end()

    Using push/pop API to define nested code ranges conveniently:

    >>> nvtx.Range("my_code_range").push()
    >>> time.sleep(1)
    >>> nvtx.Range("my_inner_code_range").push()
    >>> time.sleep(2)
    >>> nvtx.Range.pop()  # pops inner
    >>> nvtx.Range.pop()  # pops outer
    """

    def __init__(self, message=None, color="blue", domain=None):
        self._attributes = libnvtx.EventAttributes(
            message, color_to_hex(color)
        )
        self._domain = Domain(domain)

    @property
    def message(self):
        return self._attributes.message

    def start(self):
        self._id = libnvtx.range_start(self._attributes, self._domain.handle)

    def end(self):
        libnvtx.range_end(self._id, self._domain.handle)

    def push(self):
        libnvtx.range_push(self._attributes, self._domain.handle)

    @classmethod
    def pop(self):
        libnvtx.range_pop()


@contextmanager
def annotate(*args, **kwargs):
    """
    Annotate a function or a code range.

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
    rng = Range(*args, **kwargs)
    rng.push()
    try:
        yield
    finally:
        Range.pop()


_annotate = annotate


class RegisteredMessage(metaclass=CachedInstanceMeta):
    def __init__(self, message):
        print(f"Constructing RegisteredMessage({message})")
        self.message = message

    def __repr__(self):
        return f"RegisteredMessage({self.message})"
