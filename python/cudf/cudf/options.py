# Copyright (c) 2022-2025, NVIDIA CORPORATION.
from __future__ import annotations

import os
import textwrap
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import Callable, Container


@dataclass
class Option:
    default: Any
    value: Any
    description: str
    validator: Callable
    set_callback: Callable | None = None
    get_callback: Callable | None = None


_OPTIONS: dict[str, Option] = {}


def _env_get_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):
        return default


def _env_get_bool(name, default):
    env = os.getenv(name)
    if env is None:
        return default
    as_a_int = _env_get_int(name, None)
    env = env.lower().strip()
    if env == "true" or env == "on" or as_a_int:
        return True
    if env == "false" or env == "off" or as_a_int == 0:
        return False
    return default


def _register_option(
    name: str,
    default_value: Any,
    description: str,
    validator: Callable,
    set_callback: Callable | None = None,
    get_callback: Callable | None = None,
):
    """Register an option.

    Parameters
    ----------
    name : str
        The name of the option.
    default_value : Any
        The default value of the option.
    description : str
        A text description of the option.
    validator : Callable
        Called on the option value to check its validity. Should raise an
        error if the value is invalid.
    set_callback : Callable | None
        Called when setting the option value.
    get_callback : Callable | None
        Called when getting the option value.

    Raises
    ------
    BaseException
        Raised by validator if the value is invalid.
    """
    validator(default_value)
    _OPTIONS[name] = Option(
        default_value,
        default_value,
        description,
        validator,
        set_callback,
        get_callback,
    )


def get_option(name: str) -> Any:
    """Get the value of option.

    Parameters
    ----------
    key : str
        The name of the option.

    Returns
    -------
    The value of the option.

    Raises
    ------
    KeyError
        If option ``name`` does not exist.
    """
    try:
        option_obj = _OPTIONS[name]
        if option_obj.get_callback is not None:
            option_obj.get_callback()
        return option_obj.value
    except KeyError:
        raise KeyError(f'"{name}" does not exist.')


def set_option(name: str, val: Any):
    """Set the value of option.

    Parameters
    ----------
    name : str
        The name of the option.
    val : Any
        The value to set.

    Raises
    ------
    KeyError
        If option ``name`` does not exist.
    BaseException
        Raised by validator if the value is invalid.
    """
    try:
        option_obj = _OPTIONS[name]
        option_obj.validator(val)
        option_obj.value = val
        if option_obj.set_callback is not None:
            option_obj.set_callback()
    except KeyError:
        raise KeyError(f'"{name}" does not exist.')


def _build_option_description(name, opt):
    return (
        f"{name}:\n"
        f"\t{opt.description}\n"
        f"\t[Default: {opt.default}] [Current: {opt.value}]"
    )


def describe_option(name: str | None = None):
    """Prints the description of an option.

    If `name` is unspecified, prints the description of all available options.

    Parameters
    ----------
    name : Optional[str]
        The name of the option.
    """
    names = _OPTIONS.keys() if name is None else [name]
    for name in names:
        print(_build_option_description(name, _OPTIONS[name]))  # noqa: T201


def _make_contains_validator(valid_options: Container) -> Callable:
    """Return a validator that checks if a value is in `valid_options`."""

    def _validator(val):
        if val not in valid_options:
            raise ValueError(
                f"{val} is not a valid option. "
                f"Must be one of {set(valid_options)}."
            )

    return _validator


def _cow_validator(val):
    if val not in {False, True}:
        raise ValueError(
            f"{val} is not a valid option. Must be one of {{False, True}}."
        )


def _spill_validator(val):
    if val not in {False, True}:
        raise ValueError(
            f"{val} is not a valid option. Must be one of {{False, True}}."
        )


def _integer_validator(val):
    try:
        int(val)
        return True
    except ValueError:
        raise ValueError(f"{val} is not a valid option. Must be an integer.")


def _integer_and_none_validator(val):
    try:
        if val is None or int(val):
            return
    except ValueError:
        raise ValueError(
            f"{val} is not a valid option. Must be an integer or None."
        )


def _positive_integer_validator(val):
    if not (isinstance(val, int) and val > 0):
        raise ValueError(
            f"{val} is not a valid option. Must be a positive integer."
        )


_register_option(
    "default_integer_bitwidth",
    None,
    textwrap.dedent(
        """
        Default bitwidth when the dtype of an integer needs to be
        inferred. If set to `None`, the API will align dtype with pandas.
        APIs that respect this option include:
        \t- cudf object constructors
        \t- cudf.read_csv and cudf.read_json when `dtype` is not specified.
        \t- APIs that require implicit conversion of cudf.RangeIndex to an
        \t  integer index.
        \tValid values are  None, 32 or 64. Default is None.
    """
    ),
    _make_contains_validator([None, 32, 64]),
)


_register_option(
    "default_float_bitwidth",
    None,
    textwrap.dedent(
        """
        Default bitwidth when the dtype of a float needs to be
        inferred. If set to `None`, the API will align dtype with pandas.
        APIs that respect this option include:
        \t- cudf object constructors
        \t- cudf.read_csv and cudf.read_json when `dtype` is not specified.
        \tValid values are None, 32 or 64. Default is None.
    """
    ),
    _make_contains_validator([None, 32, 64]),
)

_register_option(
    "spill",
    _env_get_bool("CUDF_SPILL", False),
    textwrap.dedent(
        """
        Enables spilling.
        \tValid values are True or False. Default is False.
        """
    ),
    _spill_validator,
)


_register_option(
    "copy_on_write",
    _env_get_bool("CUDF_COPY_ON_WRITE", False),
    textwrap.dedent(
        """
        If set to `False`, disables copy-on-write.
        If set to `True`, enables copy-on-write.
        Read more at: :ref:`copy-on-write-user-doc`
        \tValid values are True or False. Default is False.
    """
    ),
    _cow_validator,
)


_register_option(
    "spill_on_demand",
    _env_get_bool("CUDF_SPILL_ON_DEMAND", True),
    textwrap.dedent(
        """
        Enables spilling on demand using an RMM out-of-memory error handler.
        This has no effect if spilling is disabled, see the "spill" option.
        \tValid values are True or False. Default is True.
        """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "spill_device_limit",
    _env_get_int("CUDF_SPILL_DEVICE_LIMIT", None),
    textwrap.dedent(
        """
        Enforce a device memory limit in bytes.
        This has no effect if spilling is disabled, see the "spill" option.
        \tValid values are any positive integer or None (disabled).
        \tDefault is None.
        """
    ),
    _integer_and_none_validator,
)

_register_option(
    "spill_stats",
    _env_get_int("CUDF_SPILL_STATS", 0),
    textwrap.dedent(
        """
        If not 0, enables statistics at the specified level:
            0  - disabled (no overhead).
            1+ - duration and number of bytes spilled (very low overhead).
            2+ - a traceback for each time a spillable buffer is exposed
                permanently (potential high overhead).

        Valid values are any positive integer.
        Default is 0 (disabled).
        """
    ),
    _integer_validator,
)

_register_option(
    "mode.pandas_compatible",
    False,
    textwrap.dedent(
        """
        If set to `False`, retains `cudf` specific behavior.
        If set to `True`, enables pandas compatibility mode,
        which will try to match pandas API behaviors in case of
        any inconsistency.
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "memory_profiling",
    _env_get_bool("CUDF_MEMORY_PROFILING", False),
    textwrap.dedent(
        """
        If set to `False`, disables memory profiling.
        If set to `True`, enables memory profiling.
        Read more at: :ref:`memory-profiling-user-doc`
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "io.parquet.low_memory",
    False,
    textwrap.dedent(
        """
        If set to `False`, reads entire parquet in one go.
        If set to `True`, reads parquet file in chunks.
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "io.json.low_memory",
    False,
    textwrap.dedent(
        """
        If set to `False`, reads entire json in one go.
        If set to `True`, reads json file in chunks.
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)

_register_option(
    "kvikio_remote_io",
    _env_get_bool("CUDF_KVIKIO_REMOTE_IO", False),
    textwrap.dedent(
        """
        Whether to use KvikIO's remote IO backend or not.
        \tWARN: this is experimental and may be removed at any time
        \twithout warning or deprecation period.
        \tSet KVIKIO_NTHREADS (default is 8) to change the number of
        \tconcurrent tcp connections, which is important for good performance.
        \tValid values are True or False. Default is False.
    """
    ),
    _make_contains_validator([False, True]),
)


def _num_io_threads_set_callback():
    plc.io.kvikio_manager.set_num_io_threads(_OPTIONS["io.num_threads"].value)


def _num_io_threads_get_callback():
    actual_result = plc.io.kvikio_manager.num_io_threads()
    expected_result = _OPTIONS["io.num_threads"].value
    assert actual_result == expected_result, (
        f"Mismatch in num_io_threads: Expected: {expected_result}, Actual: {actual_result}"
    )


_register_option(
    "io.num_threads",
    plc.io.kvikio_manager.num_io_threads(),
    textwrap.dedent(
        """
        The number of IO threads used by the KvikIO library.
        There are different ways to set this value. In descending order of
        override priority:
            - cuDF option API
                - `cudf.set_option("io.num_threads", value)` to set the value
                globally
                - `cudf.option_context("io.num_threads", value)` to set the value
                temporarily
            - Environment variable `KVIKIO_NTHREADS`
            - cuDF's default number of IO threads
                cuDF uses a platform-dependent default if nothing is specified
                Query this default using:
                `pylibcudf.io.kvikio_manager.default_num_io_threads()`
        \tValid values are integers. Default is a platform-dependent value.
    """
    ),
    _positive_integer_validator,
    _num_io_threads_set_callback,
    _num_io_threads_get_callback,
)


class option_context(ContextDecorator):
    """
    Context manager to temporarily set options in the `with` statement context.

    You need to invoke as ``option_context(pat, val, [(pat, val), ...])``.


    Examples
    --------
    >>> from cudf import option_context
    >>> with option_context('mode.pandas_compatible', True, 'default_float_bitwidth', 32):
    ...     pass
    """

    def __init__(self, *args) -> None:
        if len(args) % 2 != 0:
            raise ValueError(
                "Need to invoke as option_context(pat, val, "
                "[(pat, val), ...])."
            )

        self.ops = tuple(zip(args[::2], args[1::2]))

    def __enter__(self) -> None:
        self.undo = tuple((pat, get_option(pat)) for pat, _ in self.ops)
        for pat, val in self.ops:
            set_option(pat, val)

    def __exit__(self, *args) -> None:
        for pat, val in self.undo:
            set_option(pat, val)
