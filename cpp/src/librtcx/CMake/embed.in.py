# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
from typing import NamedTuple

import lz4.block
import zstd


def merge_bytes_with_null_terminators(
    bytes_lists: list[bytes],
) -> tuple[bytes, list[tuple[int, int]]]:
    merged: bytes = bytes()
    ranges: list[tuple[int, int]] = []

    for byte_data in bytes_lists:
        ranges.append((len(merged), len(byte_data)))
        merged += byte_data + b"\0"

    return merged, ranges


class EmbedOutput(NamedTuple):
    cxx_header: str | None
    cxx_source: str | None
    asm_source: str | None
    bin_file_name: str | None
    bin_file_data: bytes | None
    hash: bytes


def load_file_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()


def compress_bytes(data: bytes, compression: str) -> bytes:
    assert compression in ("none", "lz4", "zstd"), "Invalid compression type"

    if compression == "none":
        return data
    elif compression == "lz4":
        return lz4.block.compress(
            data, mode="high_compression", compression=12, store_size=False
        )
    elif compression == "zstd":
        return zstd.compress(data, 22)


def generate_cxx_source_files_data(
    id: str,
    file_paths: list[str],
    dests: list[str],
    include_directories: list[str],
    compression: str,
) -> EmbedOutput:
    uncompressed_files_bytes, files_ranges = merge_bytes_with_null_terminators(
        [load_file_bytes(p) for p in file_paths]
    )

    compress = compression != "none"

    compressed_files_bytes = (
        compress_bytes(uncompressed_files_bytes, compression)
        if compress
        else None
    )

    binary_size = (
        len(compressed_files_bytes)
        if compress
        else len(uncompressed_files_bytes)
    )

    if compress:
        print(
            f"-- Compressed {id}'s binary from {len(uncompressed_files_bytes)} bytes to {len(compressed_files_bytes)} bytes (compression ratio: {len(compressed_files_bytes) / len(uncompressed_files_bytes):.2f})"
        )

    merged_dests_bytes, _ = merge_bytes_with_null_terminators(
        [d.encode("utf-8") for d in dests]
    )

    merged_include_directories_bytes, _ = merge_bytes_with_null_terminators(
        [d.encode("utf-8") for d in include_directories]
    )

    # compute combined sha256 hash of all files
    sha = hashlib.sha256()
    sha.update(uncompressed_files_bytes)
    sha.update(merged_dests_bytes)
    sha.update(merged_include_directories_bytes)
    sha.update(compression.encode("utf-8"))

    hash: bytes = sha.digest()

    binary_file_name = f"embed_{id}.bin"

    include_dirs_list = ",\n".join([f'"{d}"' for d in include_directories])
    dests_list = ",\n".join([f'"{d}"' for d in dests])
    ranges_list = ",\n".join(
        [f"{{{offset}, {size}}}" for offset, size in files_ranges]
    )
    hash_list = ", ".join([f"0x{b:02x}" for b in hash])

    cxx_header = f"""
// Auto-generated header for embedded files with ID: {id}
#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <string_view>

namespace rtcx_embed {{

struct range {{
  std::size_t offset = 0;
  std::size_t size   = 0;
}};

constexpr char const * {id}_include_directories[{len(include_directories)}] =
{{
{include_dirs_list}
}};

constexpr char const * {id}_file_destinations[{len(dests)}] =
{{
{dests_list}
}};

constexpr range {id}_file_ranges[{len(files_ranges)}] =
{{
{ranges_list}
}};

constexpr std::size_t {id}_files_uncompressed_size = {len(uncompressed_files_bytes)};

constexpr char const * {id}_files_compression = "{compression}";

extern "C" std::uint8_t const rtcx_embed_{id}_files_begin[];

static std::span<std::uint8_t const> const {id}_files =
{{
rtcx_embed_{id}_files_begin,
{binary_size}L
}};

constexpr std::uint8_t {id}_hash[{len(hash)}] =
{{
{hash_list}
}};

}}
"""

    asm_source = f"""
.section .rodata
.global rtcx_embed_{id}_files_begin
rtcx_embed_{id}_files_begin:
.incbin "{binary_file_name}"

.section .note.GNU-stack,"",@progbits
"""

    return EmbedOutput(
        cxx_header=cxx_header,
        cxx_source=None,
        asm_source=asm_source,
        bin_file_name=binary_file_name,
        bin_file_data=compressed_files_bytes
        if compress
        else uncompressed_files_bytes,
        hash=hash,
    )


def generate_embed(
    id: str,
    file_paths: list[str],
    file_dests: list[str],
    include_directories: list[str],
    compression: str,
    output_dir: str,
):
    output = generate_cxx_source_files_data(
        id, file_paths, file_dests, include_directories, compression
    )

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/{id}.hpp", "w") as f:
        f.write(output.cxx_header if output.cxx_header is not None else "")

    with open(f"{output_dir}/{id}.s", "w") as f:
        f.write(output.asm_source if output.asm_source is not None else "")

    if output.bin_file_name and output.bin_file_data:
        with open(f"{output_dir}/{output.bin_file_name}", "wb") as f:
            f.write(output.bin_file_data)


def main():
    id: str = "@RTCX_EMBED_PY_ARG__ID@"
    file_paths: list[str] = "@RTCX_EMBED_PY_ARG__FILE_PATHS@".split(";")
    file_dests: list[str] = "@RTCX_EMBED_PY_ARG__FILE_DESTS@".split(";")
    include_directories: list[str] = "@RTCX_EMBED_PY_ARG__INCLUDE_DIRS@".split(
        ";"
    )
    compression: str = "@RTCX_EMBED_PY_ARG__COMPRESSION@"
    output_dir: str = "@RTCX_EMBED_PY_ARG__OUTPUT_DIR@"

    generate_embed(
        id,
        file_paths,
        file_dests,
        include_directories,
        compression,
        output_dir,
    )


if __name__ == "__main__":
    main()
