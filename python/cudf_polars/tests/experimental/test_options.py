# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for StreamingOptions."""

from __future__ import annotations

import argparse
import os

import pytest

from cudf_polars.experimental.rapidsmpf.frontend.options import (
    UNSPECIFIED,
    StreamingOptions,
    _Unspecified,
)

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------


def test_unspecified_is_singleton() -> None:
    assert _Unspecified() is UNSPECIFIED


def test_unspecified_repr() -> None:
    assert repr(UNSPECIFIED) == "UNSPECIFIED"


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------


def test_all_fields_unspecified_by_default() -> None:
    opts = StreamingOptions()
    assert isinstance(opts.fallback_mode, _Unspecified)
    assert isinstance(opts.log, _Unspecified)
    assert isinstance(opts.raise_on_fail, _Unspecified)


# ---------------------------------------------------------------------------
# to_executor_options
# ---------------------------------------------------------------------------


def test_executor_options_empty_when_all_unspecified() -> None:
    assert StreamingOptions().to_executor_options() == {}


def test_executor_options_includes_set_fields() -> None:
    opts = StreamingOptions(fallback_mode="raise", max_rows_per_partition=500_000)
    result = opts.to_executor_options()
    assert result["fallback_mode"] == "raise"
    assert result["max_rows_per_partition"] == 500_000
    assert "log" not in result


def test_executor_options_unique_fraction() -> None:
    result = StreamingOptions(unique_fraction={"col_a": 0.5}).to_executor_options()
    assert result["unique_fraction"] == {"col_a": 0.5}


def test_executor_options_num_py_executors() -> None:
    result = StreamingOptions(num_py_executors=4).to_executor_options()
    assert result["num_py_executors"] == 4


# ---------------------------------------------------------------------------
# to_engine_options
# ---------------------------------------------------------------------------


def test_engine_options_empty_when_all_unspecified() -> None:
    assert StreamingOptions().to_engine_options() == {}


def test_engine_options_includes_set_fields() -> None:
    result = StreamingOptions(
        raise_on_fail=True, cuda_stream_policy="pool"
    ).to_engine_options()
    assert result["raise_on_fail"] is True
    assert result["cuda_stream_policy"] == "pool"
    assert "log" not in result


# ---------------------------------------------------------------------------
# to_rapidsmpf_options
# ---------------------------------------------------------------------------


def test_rapidsmpf_options_serialized() -> None:
    opts = StreamingOptions(
        statistics=True, pinned_memory=False, num_streaming_threads=8, log="DEBUG"
    )
    strings = opts.to_rapidsmpf_options().get_strings()
    assert strings["statistics"] == "True"
    assert strings["pinned_memory"] == "False"
    assert strings["num_streaming_threads"] == "8"
    assert strings["log"] == "DEBUG"


def test_rapidsmpf_options_unspecified_fields_absent() -> None:
    env_backup = {
        k: os.environ.pop(k) for k in list(os.environ) if k.startswith("RAPIDSMPF_")
    }
    try:
        assert StreamingOptions().to_rapidsmpf_options().get_strings() == {}
    finally:
        os.environ.update(env_backup)


def test_rapidsmpf_options_picks_up_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPIDSMPF_LOG", "TRACE")
    assert StreamingOptions().to_rapidsmpf_options().get_strings()["log"] == "TRACE"


def test_rapidsmpf_options_explicit_overrides_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAPIDSMPF_LOG", "TRACE")
    strings = StreamingOptions(log="WARN").to_rapidsmpf_options().get_strings()
    assert strings["log"] == "WARN"


def test_rapidsmpf_options_env_var_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAPIDSMPF_LOG", raising=False)
    assert "log" not in StreamingOptions().to_rapidsmpf_options().get_strings()


def test_spmd_engine_from_options_creates_engine() -> None:
    """from_options with default StreamingOptions creates a valid SPMDEngine."""
    pytest.importorskip("rapidsmpf")
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    opts = StreamingOptions(fallback_mode="silent", raise_on_fail=True)
    with SPMDEngine.from_options(opts) as engine:
        assert engine.nranks >= 1


test_spmd_engine_from_options_creates_engine = pytest.mark.spmd(
    test_spmd_engine_from_options_creates_engine
)


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_empty_equals_default() -> None:
    assert StreamingOptions.from_dict({}) == StreamingOptions()


def test_from_dict_maps_known_fields() -> None:
    opts = StreamingOptions.from_dict(
        {"fallback_mode": "raise", "num_streaming_threads": 8}
    )
    assert opts.fallback_mode == "raise"
    assert opts.num_streaming_threads == 8
    assert isinstance(opts.log, _Unspecified)


def test_from_dict_none_value_is_unspecified() -> None:
    opts = StreamingOptions.from_dict({"fallback_mode": None})
    assert isinstance(opts.fallback_mode, _Unspecified)


def test_from_dict_unknown_key_raises() -> None:
    with pytest.raises(TypeError):
        StreamingOptions.from_dict({"no_such_field": 42})


def test_from_dict_roundtrip() -> None:
    original = StreamingOptions(fallback_mode="silent", num_streaming_threads=4)
    reconstructed = StreamingOptions.from_dict(
        {"fallback_mode": "silent", "num_streaming_threads": 4}
    )
    assert reconstructed == original


# ---------------------------------------------------------------------------
# _from_argparse
# ---------------------------------------------------------------------------


def test_from_argparse_empty_namespace_equals_default() -> None:
    assert StreamingOptions._from_argparse(argparse.Namespace()) == StreamingOptions()


def test_from_argparse_direct_fields() -> None:
    opts = StreamingOptions._from_argparse(
        argparse.Namespace(fallback_mode="raise", max_rows_per_partition=500_000)
    )
    assert opts.fallback_mode == "raise"
    assert opts.max_rows_per_partition == 500_000


def test_from_argparse_renames() -> None:
    opts = StreamingOptions._from_argparse(
        argparse.Namespace(
            rapidsmpf_log="DEBUG",
            rapidsmpf_statistics=True,
            blocksize=1_000_000,
        )
    )
    assert opts.log == "DEBUG"
    assert opts.statistics is True
    assert opts.target_partition_size == 1_000_000


def test_from_argparse_dynamic_planning() -> None:
    assert isinstance(
        StreamingOptions._from_argparse(
            argparse.Namespace(dynamic_planning=True)
        ).dynamic_planning,
        _Unspecified,
    )
    assert (
        StreamingOptions._from_argparse(
            argparse.Namespace(dynamic_planning=False)
        ).dynamic_planning
        is None
    )


def test_from_argparse_stream_policy() -> None:
    assert isinstance(
        StreamingOptions._from_argparse(
            argparse.Namespace(stream_policy="auto")
        ).cuda_stream_policy,
        _Unspecified,
    )
    assert (
        StreamingOptions._from_argparse(
            argparse.Namespace(stream_policy="pool")
        ).cuda_stream_policy
        == "pool"
    )


# ---------------------------------------------------------------------------
# _add_cli_args
# ---------------------------------------------------------------------------


def test_add_cli_args_then_from_argparse_roundtrip() -> None:
    parser = argparse.ArgumentParser()
    StreamingOptions._add_cli_args(parser)
    args = parser.parse_args(
        ["--num-streaming-threads", "8", "--rapidsmpf-log", "DEBUG", "--raise-on-fail"]
    )
    opts = StreamingOptions._from_argparse(args)
    assert opts.num_streaming_threads == 8
    assert opts.log == "DEBUG"
    assert opts.raise_on_fail is True
    # Unprovided args default to None → UNSPECIFIED
    assert isinstance(opts.fallback_mode, _Unspecified)


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, -100])
def test_num_streaming_threads_invalid(value: int) -> None:
    with pytest.raises(ValueError, match="num_streaming_threads must be > 0"):
        StreamingOptions(num_streaming_threads=value)


@pytest.mark.parametrize("value", [0, -1, -100])
def test_num_streams_invalid(value: int) -> None:
    with pytest.raises(ValueError, match="num_streams must be > 0"):
        StreamingOptions(num_streams=value)


@pytest.mark.parametrize("value", ["ERROR", "VERBOSE", "debug", ""])
def test_log_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="log must be one of"):
        StreamingOptions(log=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["WARN", "ignore", ""])
def test_fallback_mode_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="fallback_mode must be one of"):
        StreamingOptions(fallback_mode=value)


@pytest.mark.parametrize("value", [1, 8, 64])
def test_num_streaming_threads_valid(value: int) -> None:
    opts = StreamingOptions(num_streaming_threads=value)
    assert opts.num_streaming_threads == value


@pytest.mark.parametrize("log", ["NONE", "PRINT", "WARN", "INFO", "DEBUG", "TRACE"])
def test_log_valid_values(log: str) -> None:
    opts = StreamingOptions(log=log)  # type: ignore[arg-type]
    assert opts.log == log


@pytest.mark.parametrize("mode", ["warn", "raise", "silent"])
def test_fallback_mode_valid_values(mode: str) -> None:
    opts = StreamingOptions(fallback_mode=mode)
    assert opts.fallback_mode == mode


# ---------------------------------------------------------------------------
# to_dict / roundtrip
# ---------------------------------------------------------------------------


def test_to_dict_empty_when_all_unspecified() -> None:
    assert StreamingOptions().to_dict() == {}


def test_to_dict_contains_only_set_fields() -> None:
    opts = StreamingOptions(fallback_mode="silent", num_streaming_threads=4)
    d = opts.to_dict()
    assert d == {"fallback_mode": "silent", "num_streaming_threads": 4}


def test_to_dict_roundtrip() -> None:
    original = StreamingOptions(
        fallback_mode="silent",
        num_streaming_threads=4,
        log="DEBUG",
        raise_on_fail=True,
    )
    assert StreamingOptions.from_dict(original.to_dict()) == original


def test_to_dict_roundtrip_empty() -> None:
    opts = StreamingOptions()
    assert StreamingOptions.from_dict(opts.to_dict()) == opts
