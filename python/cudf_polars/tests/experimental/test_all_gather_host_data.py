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


def test_per_rank_data(engine) -> None:
    """Each rank sends its rank index; results are ordered by rank."""
    comm = engine.comm
    br = engine.context.br()
    data = struct.pack("i", comm.rank)
    result = all_gather_host_data(comm, br, op_id=4, data=data)
    assert len(result) == comm.nranks
    for i, item in enumerate(result):
        assert struct.unpack("i", item)[0] == i


def test_empty_bytes(engine) -> None:
    """Gathering empty bytes from every rank works."""
    comm = engine.comm
    br = engine.context.br()
    result = all_gather_host_data(comm, br, op_id=1, data=b"")
    assert len(result) == comm.nranks
    for item in result:
        assert item == b""


def test_structured_data(engine) -> None:
    """Struct-packed integers survive the gather round-trip."""
    comm = engine.comm
    br = engine.context.br()
    value = 12345678901234
    data = struct.pack("q", value)
    result = all_gather_host_data(comm, br, op_id=2, data=data)
    assert len(result) == comm.nranks
    for item in result:
        assert struct.unpack("q", item)[0] == value


def test_bytearray_input(engine) -> None:
    """A bytearray input is accepted and gathered correctly."""
    comm = engine.comm
    br = engine.context.br()
    data = bytearray(b"bytearray input")
    result = all_gather_host_data(comm, br, op_id=3, data=data)
    assert len(result) == comm.nranks
    for item in result:
        assert item == bytes(data)


def test_gather_cluster_info(engine) -> None:
    """SPMDEngine.gather_cluster_info returns ClusterInfo for each rank."""
    infos = engine.gather_cluster_info()
    assert len(infos) == engine.nranks
    for info in infos:
        assert isinstance(info, ClusterInfo)
        assert info.pid > 0
        assert isinstance(info.hostname, str)
        assert info.cuda_visible_devices is None or isinstance(
            info.cuda_visible_devices, str
        )
        assert isinstance(info.gpu_uuid, str)
    # Each rank runs in its own process.
    assert len({info.pid for info in infos}) == engine.nranks
    # Without allow_gpu_sharing, all UUIDs must be unique (enforced at init).
    assert len({info.gpu_uuid for info in infos}) == engine.nranks


def test_cluster_info_cuda_visible_devices(monkeypatch) -> None:
    """ClusterInfo.local() picks up CUDA_VISIBLE_DEVICES from the environment."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    info = ClusterInfo.local()
    assert info.cuda_visible_devices == "3,7"


def test_cluster_info_cuda_visible_devices_unset(monkeypatch) -> None:
    """ClusterInfo.local() returns None when CUDA_VISIBLE_DEVICES is not set."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    info = ClusterInfo.local()
    assert info.cuda_visible_devices is None


@pytest.mark.parametrize(
    "engine",
    [{"engine_options": {"allow_gpu_sharing": True}}],
    indirect=True,
)
def test_allow_gpu_sharing(engine) -> None:
    """Engine init succeeds with allow_gpu_sharing=True."""
    assert engine.nranks >= 1
