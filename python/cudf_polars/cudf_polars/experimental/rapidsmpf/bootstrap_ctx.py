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


def is_running_under_slurm() -> bool:
    """
    Check if running under Slurm.

    Returns
    -------
    bool
        True if Slurm environment variables are detected.
    """
    # Check for Slurm environment variables
    return (
        "SLURM_JOB_ID" in os.environ
        or "SLURM_PROCID" in os.environ
        or "PMIX_NAMESPACE" in os.environ
    )


def _detect_backend_type():
    """
    Detect the appropriate backend type based on environment.

    Returns
    -------
    BackendType
        The detected backend type.

    Notes
    -----
    Detection logic:
    1. If RAPIDSMPF_COORD_DIR is set -> use AUTO (will choose FILE)
    2. If running under Slurm without COORD_DIR -> set up FILE backend with temp dir
    3. Otherwise use AUTO (default)
    """
    if not BOOTSTRAP_AVAILABLE:
        raise RuntimeError("rapidsmpf.bootstrap not available")

    # If COORD_DIR is set, FILE backend can work - use AUTO
    if "RAPIDSMPF_COORD_DIR" in os.environ:
        return bootstrap.BackendType.AUTO

    # If running under Slurm without COORD_DIR, we need to set one up
    # because the SLURM backend is not available in Python bindings yet
    if is_running_under_slurm():
        # Check if SLURM backend is available in the Python bindings
        if hasattr(bootstrap.BackendType, "SLURM"):
            print(
                f"[Rank {get_rank()}] Detected Slurm environment, using SLURM backend",
                flush=True,
            )
            return bootstrap.BackendType.SLURM
        else:
            # SLURM backend not available in Python bindings
            # Create a temporary coordination directory as workaround

            # Use a shared temp directory based on Slurm job ID
            job_id = os.environ.get("SLURM_JOB_ID", "unknown")
            coord_dir = f"/tmp/rapidsmpf-coord-{job_id}"

            # All ranks create the directory (exist_ok=True avoids race)
            rank = get_rank()
            os.makedirs(coord_dir, exist_ok=True)

            if rank == 0:
                print(
                    f"[Rank {rank}] SLURM backend not available in Python bindings, "
                    f"using FILE backend with coordination directory: {coord_dir}",
                    flush=True,
                )
                print(
                    f"[Rank {rank}] NOTE: Clean up {coord_dir} after job completes "
                    "(or it will be reused on next job with same ID)",
                    flush=True,
                )

            # Set the environment variable for all ranks
            os.environ["RAPIDSMPF_COORD_DIR"] = coord_dir

            return bootstrap.BackendType.AUTO

    # Default to AUTO
    return bootstrap.BackendType.AUTO


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

        # Detect and use appropriate backend
        backend_type = _detect_backend_type()

        # Debug: print environment info on rank 0
        rank = get_rank()
        if rank == 0:
            print(f"[Bootstrap] Backend type: {backend_type}", flush=True)
            print(
                f"[Bootstrap] RAPIDSMPF_COORD_DIR: {os.environ.get('RAPIDSMPF_COORD_DIR', 'NOT SET')}",
                flush=True,
            )
            print(
                f"[Bootstrap] SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'NOT SET')}",
                flush=True,
            )
            print(
                f"[Bootstrap] PMIX_NAMESPACE: {os.environ.get('PMIX_NAMESPACE', 'NOT SET')}",
                flush=True,
            )

        try:
            # Initialize the bootstrap context with detected backend
            _global_context = bootstrap.create_ucxx_comm(backend_type)
        except RuntimeError as e:
            # Provide helpful error message
            error_msg = f"Failed to initialize bootstrap context: {e}\n"
            error_msg += "\nEnvironment variables:\n"
            error_msg += f"  RAPIDSMPF_RANK: {os.environ.get('RAPIDSMPF_RANK', 'NOT SET')}\n"
            error_msg += (
                f"  RAPIDSMPF_NRANKS: {os.environ.get('RAPIDSMPF_NRANKS', 'NOT SET')}\n"
            )
            error_msg += f"  RAPIDSMPF_COORD_DIR: {os.environ.get('RAPIDSMPF_COORD_DIR', 'NOT SET')}\n"
            error_msg += f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'NOT SET')}\n"
            error_msg += (
                f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'NOT SET')}\n"
            )
            error_msg += f"  SLURM_NPROCS: {os.environ.get('SLURM_NPROCS', 'NOT SET')}\n"
            error_msg += (
                f"  PMIX_NAMESPACE: {os.environ.get('PMIX_NAMESPACE', 'NOT SET')}\n"
            )
            error_msg += "\nFor Slurm, ensure you're using: srun --mpi=pmix ...\n"
            error_msg += (
                "For rrun, ensure RAPIDSMPF_COORD_DIR is set or rrun is launching.\n"
            )
            raise RuntimeError(error_msg) from e

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
    # Try RAPIDSMPF_RANK first, fall back to SLURM_PROCID for Slurm
    rank = os.environ.get("RAPIDSMPF_RANK")
    if rank is not None:
        return int(rank)
    # Fall back to Slurm env vars
    rank = os.environ.get("SLURM_PROCID")
    if rank is not None:
        return int(rank)
    return 0


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
    # Try RAPIDSMPF_NRANKS first, fall back to SLURM_NPROCS/SLURM_NTASKS for Slurm
    nranks = os.environ.get("RAPIDSMPF_NRANKS")
    if nranks is not None:
        return int(nranks)
    # Fall back to Slurm env vars
    nranks = os.environ.get("SLURM_NPROCS") or os.environ.get("SLURM_NTASKS")
    if nranks is not None:
        return int(nranks)
    return 1
