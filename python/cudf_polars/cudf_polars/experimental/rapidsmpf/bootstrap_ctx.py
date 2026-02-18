# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap context management for rrun execution."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapidsmpf.bootstrap.bootstrap import Context
    from rapidsmpf.integrations import WorkerContext

try:
    import rapidsmpf.bootstrap as bootstrap

    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False

_global_context: Context | None = None
_global_worker_context: WorkerContext | None = None


class _RrunWorker:
    """Sentinel object representing the rrun worker process for rmpf_worker_setup."""

    def __init__(self, rank: int) -> None:
        self._rank = rank

    def __str__(self) -> str:
        return f"rrun-rank-{self._rank}"


def is_running_with_rrun() -> bool:
    """
    Check if running with rrun.

    Returns
    -------
    bool
        True if the RAPIDSMPF_RANK environment variable is set,
        indicating execution under rrun.
    """
    if not BOOTSTRAP_AVAILABLE:
        return False
    return bootstrap.is_running_with_rrun()


def is_running_with_slurm() -> bool:
    """
    Check if running with Slurm.

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
    The necessity of this function should be revisited. In an ideal case
    BackendType.AUTO should suffice.

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
    if is_running_with_slurm():
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
    Get or initialize bootstrap context.

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

        backend_type = _detect_backend_type()

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

    Raises
    ------
    RuntimeError
        If not running with rrun or rank could not be determined.
    """
    if not is_running_with_rrun():
        raise RuntimeError("Not running with rrun.")
    # Read directly from environment variable for efficiency
    # Try RAPIDSMPF_RANK first, fall back to SLURM_PROCID for Slurm
    rank = os.environ.get("RAPIDSMPF_RANK")
    if rank is not None:
        return int(rank)
    # Fall back to Slurm env vars
    rank = os.environ.get("SLURM_PROCID")
    if rank is not None:
        return int(rank)
    raise RuntimeError("Could not determine rank.")


def get_nranks() -> int:
    """
    Get total number of ranks.

    Returns
    -------
    int
        The total number of ranks (1 if not running under rrun).

    Raises
    ------
    RuntimeError
        If not running with rrun or number of ranks could not be determined.
    """
    return bootstrap.get_nranks()


def setup_rrun_worker_context(
    *,
    spill_device: float = 0.5,
    spill_to_pinned_memory: bool = False,
    oom_protection: bool = False,
    max_io_threads: int = 2,
) -> WorkerContext:
    """
    Initialize the rrun worker context once.

    This calls ``rmpf_worker_setup()`` to create a ``WorkerContext`` with
    properly configured ``BufferResource``, ``ProgressThread``, ``Statistics``,
    and spill functions, matching the one-time setup that Dask performs via
    ``bootstrap_dask_cluster()`` and ``dask_worker_setup()``.

    Parameters
    ----------
    spill_device
        Device memory threshold for spilling (fraction, 0.0-1.0).
    spill_to_pinned_memory
        Whether to spill to pinned host memory.
    oom_protection
        Whether to use managed memory fallback for OOM protection.
    max_io_threads
        Maximum number of IO threads.

    Returns
    -------
    WorkerContext
        The initialized worker context.
    """
    global _global_worker_context
    if _global_worker_context is not None:
        return _global_worker_context

    from rapidsmpf.config import Options, get_environment_variables
    from rapidsmpf.integrations.core import rmpf_worker_setup

    comm = get_bootstrap_context()

    options = Options(
        {
            "rrun_spill_device": str(spill_device),
            "rrun_spill_to_pinned_memory": str(spill_to_pinned_memory),
            "rrun_oom_protection": str(oom_protection),
            "rrun_statistics": "False",
            "rrun_print_statistics": "False",
            "num_streaming_threads": str(max(max_io_threads, 1)),
        }
        | get_environment_variables()
    )

    rank = get_rank()
    worker = _RrunWorker(rank)
    _global_worker_context = rmpf_worker_setup(
        worker, "rrun_", comm=comm, options=options
    )

    if rank == 0:
        print("[rrun] Worker context initialized", flush=True)

    return _global_worker_context


def get_rrun_worker_context() -> WorkerContext:
    """
    Get the initialized rrun worker context.

    Returns
    -------
    WorkerContext
        The rrun worker context.

    Raises
    ------
    RuntimeError
        If the worker context has not been initialized yet.
    """
    if _global_worker_context is None:
        raise RuntimeError(
            "rrun worker context not initialized. "
            "Call setup_rrun_worker_context() first."
        )
    return _global_worker_context
