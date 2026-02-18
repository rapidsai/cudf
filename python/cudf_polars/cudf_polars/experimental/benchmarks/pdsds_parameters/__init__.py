# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""PDS-DS query parameters for all scale factors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PARAMS_FILE = Path(__file__).parent / "parameter_substitutions.json"
_PARAMS_CACHE = None


def load_parameters(scale_factor: int, query_id: int) -> dict[str, Any]:
    """
    Load parameters for a specific query and scale factor.

    Parameters
    ----------
    scale_factor : int
        The PDS-DS scale factor (e.g., 1, 10, 100, 1000, 10000)
    query_id : int
        The PDS-DS query number (1-11)

    Returns
    -------
    dict
        Dictionary of parameter names to values

    Raises
    ------
    ValueError
        If parameters are not found for the given scale factor or query ID
    """
    global _PARAMS_CACHE  # noqa: PLW0603

    if _PARAMS_CACHE is None:
        with _PARAMS_FILE.open() as f:
            _PARAMS_CACHE = json.load(f)

    scale_params = _PARAMS_CACHE["scale_factors"].get(str(scale_factor))
    if scale_params is None:
        msg = f"No parameters found for scale factor {scale_factor}"
        raise ValueError(msg)

    params = scale_params.get(str(query_id))
    if params is None:
        msg = f"No parameters found for query {query_id} at scale factor {scale_factor}"
        raise ValueError(msg)

    return params


__all__ = ["load_parameters"]
