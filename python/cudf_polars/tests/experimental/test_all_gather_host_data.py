# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for all_gather_host_data."""

from __future__ import annotations

import struct

import pytest

from cudf_polars.experimental.rapidsmpf.frontend.core import (
    ClusterInfo,
    all_gather_host_data,
)

pytestmark = pytest.mark.spmd


def _empty(rank: int) -> bytes:
    return b""


def _text(rank: int) -> bytes:
    return f"hello from rank {rank}".encode()


def _bytearray(rank: int) -> bytearray:
    return bytearray(f"ba-rank-{rank}".encode())


def _struct(rank: int) -> bytes:
    return struct.pack("qi", 12345678901234 + rank, rank)


@pytest.mark.parametrize("make_data", [_empty, _text, _bytearray, _struct])
def test_all_gather_host_data(streaming_engine, make_data) -> None:
    """Each rank sends rank-specific data; results are correct and ordered."""
    comm = streaming_engine.comm
    br = streaming_engine.context.br()
    result = all_gather_host_data(comm, br, op_id=0, data=make_data(comm.rank))
    assert len(result) == comm.nranks
    for i, item in enumerate(result):
        assert item == bytes(make_data(i))


def test_gather_cluster_info(streaming_engine) -> None:
    """SPMDEngine.gather_cluster_info returns ClusterInfo for each rank."""
    infos = streaming_engine.gather_cluster_info()
    assert len(infos) == streaming_engine.nranks
    for info in infos:
        assert isinstance(info, ClusterInfo)
        assert info.pid > 0
        assert isinstance(info.hostname, str)
        assert info.cuda_visible_devices is None or isinstance(
            info.cuda_visible_devices, str
        )
        assert isinstance(info.gpu_uuid, str)
    # Each rank runs in its own process.
    assert len({info.pid for info in infos}) == streaming_engine.nranks
    # Without allow_gpu_sharing, all UUIDs must be unique (enforced at init).
    assert len({info.gpu_uuid for info in infos}) == streaming_engine.nranks


def test_cluster_info_cuda_visible_devices(monkeypatch) -> None:
    """ClusterInfo.local() picks up CUDA_VISIBLE_DEVICES from the environment."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    assert ClusterInfo.local().cuda_visible_devices == "3,7"


def test_cluster_info_cuda_visible_devices_unset(monkeypatch) -> None:
    """ClusterInfo.local() returns None when CUDA_VISIBLE_DEVICES is not set."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert ClusterInfo.local().cuda_visible_devices is None


@pytest.mark.parametrize(
    "streaming_engine",
    [{"engine_options": {"allow_gpu_sharing": True}}],
    indirect=True,
)
def test_allow_gpu_sharing(streaming_engine) -> None:
    """Engine init succeeds with allow_gpu_sharing=True."""
    assert streaming_engine.nranks >= 1
