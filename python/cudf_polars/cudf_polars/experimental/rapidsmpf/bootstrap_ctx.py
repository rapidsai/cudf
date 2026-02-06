# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap context management for rrun execution."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapidsmpf.bootstrap.bootstrap import Context

try:
    import rapidsmpf.bootstrap as bootstrap

    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False

_global_context: Context | None = None


def is_running_with_rrun() -> bool:
    """
    Check if running under rrun.

    Returns
    -------
    bool
        True if the RAPIDSMPF_RANK environment variable is set,
        indicating execution under rrun.
    """
    if not BOOTSTRAP_AVAILABLE:
        return False
    return bootstrap.is_running_with_rrun()


def get_bootstrap_context() -> Context:
    """
    Get or initialize bootstrap context (singleton).

    Returns
    -------
    Context
        The RapidsMPF bootstrap context.

    Raises
    ------
    RuntimeError
        If rapidsmpf.bootstrap is not available or not running under rrun.
    """
    global _global_context
    if _global_context is None:
        if not BOOTSTRAP_AVAILABLE:
            raise RuntimeError(
                "rapidsmpf.bootstrap not available. "
                "Please install rapidsmpf to use rrun execution."
            )
        if not is_running_with_rrun():
            raise RuntimeError(
                "Not running under rrun (RAPIDSMPF_RANK environment variable not set). "
                "Use 'rrun -n <nranks> python ...' to launch with rrun."
            )
        # Initialize the bootstrap context
        # The context is initialized based on environment variables set by rrun
        _global_context = bootstrap.create_ucxx_comm(bootstrap.BackendType.FILE)
    return _global_context


def get_rank() -> int:
    """
    Get current rank.

    Returns
    -------
    int
        The rank of the current process (0 if not running under rrun).
    """
    if not is_running_with_rrun():
        return 0
    # Read directly from environment variable for efficiency
    # This avoids initializing the full bootstrap context just to get rank
    return int(os.environ.get("RAPIDSMPF_RANK", "0"))


def get_nranks() -> int:
    """
    Get total number of ranks.

    Returns
    -------
    int
        The total number of ranks (1 if not running under rrun).
    """
    if not is_running_with_rrun():
        return 1
    # Read directly from environment variable for efficiency
    return int(os.environ.get("RAPIDSMPF_NRANKS", "1"))
