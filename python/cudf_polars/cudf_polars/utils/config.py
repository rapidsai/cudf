# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Config utilities."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["ConfigOptions"]


class ConfigOptions:
    """
    GPUEngine configuration-option manager.

    This is a convenience class to help manage the nested
    dictionary of user-accessible `GPUEngine` options.
    """

    __slots__ = ("_hash_value", "config_options")
    _hash_value: int
    config_options: dict[str, Any]
    """The underlying (nested) config-option dictionary."""

    def __init__(self, options: dict[str, Any]):
        self.validate(options)
        self.config_options = options

    def set(self, name: str, value: Any) -> Self:
        """
        Set a user config option.

        Nested dictionary keys should be separated by periods.
        For example::

            >>> options = options.set("parquet_options.chunked", False)

        Parameters
        ----------
        name
            Period-separated config name.
        value
            New config value.
        """
        options = config_options = copy.deepcopy(self.config_options)
        keys = name.split(".")
        for k in keys[:-1]:
            assert isinstance(options, dict)
            if k not in options:
                options[k] = {}
            options = options[k]
        options[keys[-1]] = value
        return type(self)(config_options)

    def get(self, name: str, *, default: Any = None) -> Any:
        """
        Get a user config option.

        Nested dictionary keys should be separated by periods.
        For example::

            >>> chunked = config_options.get("parquet_options.chunked")

        Parameters
        ----------
        name
            Period-separated config name.
        default
            Default return value.

        Returns
        -------
        The user-specified config value, or `default`
        if the config is not found.
        """
        options = self.config_options
        keys = name.split(".")
        for k in keys[:-1]:
            assert isinstance(options, dict)
            options = options.get(k, {})
        return options.get(keys[-1], default)

    def __hash__(self) -> int:
        """Hash a ConfigOptions object."""
        try:
            return self._hash_value
        except AttributeError:
            self._hash_value = hash(json.dumps(self.config_options))
            return self._hash_value

    @staticmethod
    def validate(config: dict) -> None:
        """
        Validate a configuration-option dictionary.

        Parameters
        ----------
        config
            GPUEngine configuration options to validate.

        Raises
        ------
        ValueError
            If the configuration contains unsupported options.
        """
        if unsupported := (
            config.keys()
            - {"raise_on_fail", "parquet_options", "executor", "executor_options"}
        ):
            raise ValueError(
                f"Engine configuration contains unsupported settings: {unsupported}"
            )
        assert {"chunked", "chunk_read_limit", "pass_read_limit"}.issuperset(
            config.get("parquet_options", {})
        )

        # Validate executor_options
        executor = config.get("executor", "pylibcudf")
        if executor == "dask-experimental":
            unsupported = config.get("executor_options", {}).keys() - {
                "max_rows_per_partition",
                "parquet_blocksize",
                "cardinality_factor",
                "groupby_n_ary",
                "broadcast_join_limit",
                "shuffle_method",
            }
        else:
            unsupported = config.get("executor_options", {}).keys()
        if unsupported:
            raise ValueError(
                f"Unsupported executor_options for {executor}: {unsupported}"
            )
