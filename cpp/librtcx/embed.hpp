/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "sha256.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <zstd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rtcx_embed {

struct size_range {
  size_t offset = 0;
  size_t size   = 0;
};

std::pair<std::vector<uint8_t>, std::vector<size_range>> merge_bytes_with_null_terminators(
  std::span<std::vector<uint8_t> const> bytes_lists)
{
  std::vector<uint8_t> merged;
  std::vector<size_range> ranges;

  for (auto& byte_data : bytes_lists) {
    ranges.push_back({merged.size(), byte_data.size()});
    merged.insert(merged.end(), byte_data.begin(), byte_data.end());
    merged.push_back(0);
  }

  return {std::move(merged), std::move(ranges)};
}

struct embed_output {
  std::string cxx_header;
  std::string cxx_source;
  std::string asm_source;
  std::string bin_file_name;
  std::vector<uint8_t> bin_file_data;
};

std::vector<uint8_t> load_file_bytes(std::string_view file_path)
{
  std::string path_str(file_path);
  std::ifstream file(path_str, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error(std::format("Failed to open file at path: {}", file_path));
  }
  auto file_size = file.tellg();
  if (file_size < 0) {
    throw std::runtime_error(
      std::format("Failed to determine size of file at path: {}", file_path));
  }
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(file_size);
  if (!file.read(reinterpret_cast<char*>(bytes.data()), file_size)) {
    throw std::runtime_error(std::format("Failed to read file at path: {}", file_path));
  }
  return bytes;
}

std::vector<uint8_t> compress_bytes(std::span<uint8_t const> bytes, std::string_view compression)
{
  if (compression != "none" && compression != "zstd") {
    throw std::invalid_argument(std::format(
      "Invalid compression type: {}. Supported values are 'none' and 'zstd'", compression));
  }

  if (compression == "none") { return std::vector<uint8_t>(bytes.begin(), bytes.end()); }

  auto const max_compressed_size = ZSTD_compressBound(bytes.size());
  std::vector<uint8_t> compressed(max_compressed_size);
  auto const compressed_size =
    ZSTD_compress(compressed.data(), compressed.size(), bytes.data(), bytes.size(), 22);

  if (ZSTD_isError(compressed_size)) {
    throw std::runtime_error(std::string("ZSTD compression failed: ") +
                             ZSTD_getErrorName(compressed_size));
  }

  compressed.resize(compressed_size);
  return compressed;
}

rtcx::sha256 compute_embed_hash(std::span<uint8_t const> uncompressed_files_bytes,
                                std::span<uint8_t const> merged_dests_bytes,
                                std::span<uint8_t const> merged_include_dirs_bytes,
                                std::string_view compression)
{
  rtcx::sha256_context ctx;
  ctx.update(uncompressed_files_bytes);
  ctx.update(merged_dests_bytes);
  ctx.update(merged_include_dirs_bytes);
  ctx.update(std::span{reinterpret_cast<const uint8_t*>(compression.data()), compression.size()});
  return ctx.finalize();
}

template <typename Container, typename Formatter>
std::string join_formatted(Container& items, std::string_view delimiter, Formatter&& formatter)
{
  std::ostringstream result;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i != 0) { result << delimiter; }
    result << formatter(items[i]);
  }
  return result.str();
}

embed_output generate_cxx_source_files_data(std::string_view id,
                                            std::span<std::string_view const> file_paths,
                                            std::span<std::string_view const> file_dsts,
                                            std::span<std::string_view const> include_dirs,
                                            std::string_view compression)
{
  std::vector<std::vector<uint8_t>> file_bytes;
  file_bytes.reserve(file_paths.size());
  for (auto const& path : file_paths) {
    file_bytes.emplace_back(load_file_bytes(path));
  }

  auto [uncompressed_files_bytes, files_ranges] = merge_bytes_with_null_terminators(file_bytes);

  auto compress = compression != "none";
  std::vector<uint8_t> compressed_files_bytes =
    compress ? compress_bytes(uncompressed_files_bytes, compression) : uncompressed_files_bytes;

  auto binary_size = compress ? compressed_files_bytes.size()
                              : static_cast<std::size_t>(uncompressed_files_bytes.size());

  if (compress) {
    std::cout << std::format(
      "-- Compressed {}'s binary from {} bytes to {} bytes (compression ratio: {:.2f})\n",
      id,
      uncompressed_files_bytes.size(),
      compressed_files_bytes.size(),
      static_cast<double>(compressed_files_bytes.size()) /
        static_cast<double>(uncompressed_files_bytes.size()));
  }

  std::vector<std::vector<uint8_t>> destination_bytes;
  destination_bytes.reserve(file_dsts.size());
  for (auto const& dest : file_dsts) {
    destination_bytes.emplace_back(dest.begin(), dest.end());
  }
  auto [merged_dests_bytes, _] = merge_bytes_with_null_terminators(destination_bytes);

  std::vector<std::vector<uint8_t>> include_directory_bytes;
  include_directory_bytes.reserve(include_dirs.size());
  for (auto const& include_directory : include_dirs) {
    include_directory_bytes.emplace_back(include_directory.begin(), include_directory.end());
  }
  auto [merged_include_dirs_bytes, __] = merge_bytes_with_null_terminators(include_directory_bytes);

  auto hash = compute_embed_hash(
    uncompressed_files_bytes, merged_dests_bytes, merged_include_dirs_bytes, compression);

  auto binary_file_name = std::format("embed_{}.bin", id);

  auto include_dirs_list =
    join_formatted(include_dirs, ",\n", [](auto s) { return std::format("\"{}\"", s); });
  auto dests_list =
    join_formatted(file_dsts, ",\n", [](auto s) { return std::format("\"{}\"", s); });
  auto ranges_list = join_formatted(
    files_ranges, ",\n", [](auto r) { return std::format("{{{}, {}}}", r.offset, r.size); });

  auto hash_list = join_formatted(
    hash, ", ", [](uint8_t byte) { return std::format("0x{:02x}", static_cast<unsigned>(byte)); });

  auto cxx_header = std::format(
    R"***(
// Auto-generated header for embedded files with ID: {}
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

constexpr char const * {}_include_directories[{}] =
{{
{}
}};

constexpr char const * {}_file_destinations[{}] =
{{
{}
}};

constexpr range {}_file_ranges[{}] =
{{
{}
}};

constexpr std::size_t {}_files_uncompressed_size = {};

constexpr char const * {}_files_compression = "{}";

extern "C" std::uint8_t const rtcx_embed_{}_files_begin[];

static std::span<std::uint8_t const> const {}_files =
{{
rtcx_embed_{}_files_begin,
{}L
}};

constexpr std::uint8_t {}_hash[{}] =
{{
{}
}};

}}
)***",
    id,
    id,
    include_dirs.size(),
    include_dirs_list,
    id,
    file_dsts.size(),
    dests_list,
    id,
    files_ranges.size(),
    ranges_list,
    id,
    uncompressed_files_bytes.size(),
    id,
    compression,
    id,
    id,
    id,
    binary_size,
    id,
    hash.size(),
    hash_list);

  auto asm_source = std::format(
    R"***(
.section .rodata
.global rtcx_embed_{}_files_begin
rtcx_embed_{}_files_begin:
.incbin "{}"

.section .note.GNU-stack,"",@progbits
)***",
    id,
    id,
    binary_file_name);

  return embed_output{
    .cxx_header    = cxx_header,
    .cxx_source    = "",
    .asm_source    = asm_source,
    .bin_file_name = binary_file_name,
    .bin_file_data = compress ? compressed_files_bytes : uncompressed_files_bytes};
}

void generate_embed(std::string_view id,
                    std::span<std::string_view const> file_paths,
                    std::span<std::string_view const> file_dsts,
                    std::span<std::string_view const> include_dirs,
                    std::string_view compression,
                    std::string_view output_directory)
{
  auto output =
    generate_cxx_source_files_data(id, file_paths, file_dsts, include_dirs, compression);

  std::filesystem::create_directories(std::filesystem::path(output_directory));

  std::ofstream header_file(std::format("{}/{}.hpp", output_directory, id));
  header_file << output.cxx_header;

  std::ofstream asm_file(std::format("{}/{}.s", output_directory, id));
  asm_file << output.asm_source;

  std::ofstream bin_file(std::format("{}/{}", output_directory, output.bin_file_name),
                         std::ios::binary);
  bin_file.write(reinterpret_cast<char const*>(output.bin_file_data.data()),
                 static_cast<std::streamsize>(output.bin_file_data.size()));
}

std::vector<std::string_view> split_string(std::string_view str, char delimiter)
{
  std::vector<std::string_view> tokens;
  std::size_t start = 0;

  while (start <= str.size()) {
    auto const pos = str.find(delimiter, start);
    if (pos == std::string_view::npos) {
      tokens.push_back(str.substr(start));
      break;
    }
    tokens.push_back(str.substr(start, pos - start));
    start = pos + 1;
  }

  return tokens;
}

}  // namespace rtcx_embed
