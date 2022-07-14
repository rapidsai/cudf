# Copyright (c) 2022, NVIDIA CORPORATION.

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union


@dataclass
class CUDFOption:
    value: Any
    description: str
    validator: Callable


_CUDF_OPTIONS: Dict[str, CUDFOption] = {}


def _register_option(
    name: str, default_value: Any, description: str, validator: Callable
):
    """Register an entry in the option dictionary.

    Parameters
    ----------
    name : str
        The name of the option. Also used as the key in the dictionary.

    default_value : Any
        The default value of the option.

    description : str
        A text description of the option.

    validator : Callable
        A function that returns ``True`` if a given value is valid for the
        option, ``False`` otherwise.
    """
    if not validator(default_value):
        raise ValueError(f"Invalid default value: {default_value}")

    _CUDF_OPTIONS[name] = CUDFOption(default_value, description, validator)


def get_option(key: str) -> Any:
    """Get the value of option.

    Parameters
    ----------
    key : str
        The name of the option.

    Returns
    -------
    The value of the option.
    """
    return _CUDF_OPTIONS[key].value


def set_option(key: str, val: Any):
    """Set the value of option.

    Raises ``ValueError`` if the provided value is invalid.

    Parameters
    ----------
    key : str
        The name of the option.
    val : Any
        The value to set.
    """
    config = _CUDF_OPTIONS[key]
    if not config.validator(val):
        raise ValueError(f"Invalid option {val}")
    config.value = val


def describe_option(key: Optional[str] = None) -> Union[str, Dict[str, str]]:
    """Returns a specific option description or all option descriptions.

    Parameters
    ----------
    key : str
        The name of the option.

    Returns
    -------
    A string description of the option or a dictionary of all option
    descriptions.
    """
    if key is None:
        return {key: _CUDF_OPTIONS[key].description for key in _CUDF_OPTIONS}

    return _CUDF_OPTIONS[key].description


_register_option(
    "default_integer_bitwidth",
    64,
    "Default integer bitwidth when inferring integer column or scalars."
    "Influences integer column bitwidth for csv, json readers if unspecified "
    "and integer column bitwidth constructed from python scalar and lists. "
    "Valid values are 32 or 64.",
    lambda x: x in [32, 64],
)
