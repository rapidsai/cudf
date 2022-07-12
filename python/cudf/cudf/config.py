# Copyright (c) 2022, NVIDIA CORPORATION.

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class CUDFConfiguration:
    name: str
    description: str
    value: Any
    validator: Callable


_CUDF_CONFIG: Dict[str, CUDFConfiguration] = {}


def register_config(
    name: str, default_value: Any, description: str, validator: Callable
):
    """Add a registry to the configuration dictionary.

    Parameters
    ----------
    name : str
        The name of the configuration. Also used as the key in the dictionary.

    default_value : Any
        The default value of the configuration.

    description : str
        A text description of the configuration.

    validator : Callable
        A function that returns ``True`` is a given value is valid for the
        configuration, ``False`` otherwise.
    """
    if not validator(default_value):
        raise ValueError(f"Invalid default value: {default_value}")

    _CUDF_CONFIG[name] = CUDFConfiguration(
        name, default_value, description, validator
    )


def get_config(key: str) -> Any:
    """Get the value of configuration.

    Parameters
    ----------
    key : str
        The name of the configuration.

    Returns
    -------
    The value of the configuration.
    """
    return _CUDF_CONFIG[key].value


def set_config(key: str, val: Any):
    """Set the value of configuration.

    Raises ``ValueError`` if val is invalid to the configuration.

    Parameters
    ----------
    key : str
        The name of the configuration.
    val : Any
        The value to set.
    """
    config = _CUDF_CONFIG[key]
    if not config.validator(val):
        raise ValueError(f"Invalid configuration {val}")
    config.value = val


def describe_config(key: str) -> str:
    """Returns the description of the configuration.

    Parameters
    ----------
    key : str
        The name of the configuration.
    """
    return _CUDF_CONFIG[key].description


def describe_configs() -> Dict[str, str]:
    """Returns all descriptions available in cudf.

    Returns
    -------
    descriptions : Dict[str, str]
        A dictionary mapping the name of the config to their descriptions.
    """
    return {
        config.name: config.description for config in _CUDF_CONFIG.values()
    }
