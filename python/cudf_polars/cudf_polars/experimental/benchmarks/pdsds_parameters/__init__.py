# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""PDS-DS query parameters for all scale factors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PARAMS_FILE = Path(__file__).parent / "parameter_substitutions.json"
_PARAMS_CACHE = None


def load_parameters(
    scale_factor: int | str, query_id: int, *, qualification: bool = False
) -> dict[str, Any]:
    """
    Load parameters for a specific query and scale factor.

    Parameters
    ----------
    scale_factor : int | str
        The PDS-DS scale factor (e.g., 1, 10, 100, 1000, 10000) or "qualification"
        for TPC-DS qualification parameters. Ignored if qualification=True.
    query_id : int
        The PDS-DS query number
    qualification : bool
        If True, load TPC-DS qualification parameters from specification Appendix B
        instead of randomly generated parameters for the given scale factor.

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

    sf_key = "qualification" if qualification else str(scale_factor)

    scale_params = _PARAMS_CACHE["scale_factors"].get(sf_key)
    if scale_params is None:
        if qualification:
            msg = "No qualification parameters found"
        else:
            msg = f"No parameters found for scale factor {scale_factor}"
        raise ValueError(msg)

    params = scale_params.get(str(query_id))
    if params is None:
        param_type = (
            "qualification" if qualification else f"scale factor {scale_factor}"
        )
        msg = f"No parameters found for query {query_id} at {param_type}"
        raise ValueError(msg)

    return params


__all__ = ["load_parameters"]
