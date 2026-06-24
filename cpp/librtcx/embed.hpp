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
#include <map>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define RTCX_EMBED_EXPECTS(condition, message)                                         \
  do {                                                                                 \
    if (!(condition)) {                                                                \
      throw std::runtime_error(std::format("{}:{}: {}", __FILE__, __LINE__, message)); \
    }                                                                                  \
  } while (false)

namespace rtcx_embed {

struct size_range {
  size_t offset = 0;
  size_t size   = 0;
};

struct embed_output {
  std::string cxx_header;
  std::string asm_source;
  std::vector<uint8_t> bin_file_data;
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

std::vector<uint8_t> load_file_bytes(std::string_view file_path)
{
  std::string path_str(file_path);
  std::ifstream file(path_str, std::ios::binary | std::ios::ate);
  RTCX_EMBED_EXPECTS(file.is_open(), std::format("Failed to open file at path: {}", file_path));
  auto file_size = file.tellg();
  RTCX_EMBED_EXPECTS(file_size >= 0,
                     std::format("Failed to determine size of file at path: {}", file_path));
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(file_size);
  RTCX_EMBED_EXPECTS(file.read(reinterpret_cast<char*>(bytes.data()), file_size),
                     std::format("Failed to read file at path: {}", file_path));
  return bytes;
}

std::vector<uint8_t> compress_bytes(std::span<uint8_t const> bytes, std::string_view compression)
{
  RTCX_EMBED_EXPECTS(
    compression == "none" || compression == "zstd",
    std::format("Invalid compression type: {}. Supported values are 'none' and 'zstd'",
                compression));

  if (compression == "none") { return std::vector<uint8_t>(bytes.begin(), bytes.end()); }

  auto const max_compressed_size = ZSTD_compressBound(bytes.size());
  std::vector<uint8_t> compressed(max_compressed_size);
  auto const compressed_size =
    ZSTD_compress(compressed.data(), compressed.size(), bytes.data(), bytes.size(), 22);

  RTCX_EMBED_EXPECTS(
    !ZSTD_isError(compressed_size),
    std::format("Compression failed with error: {}", ZSTD_getErrorName(compressed_size)));

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

enum class value_type : int8_t { INT, STRING };

std::string generate_arrays(std::span<std::string_view const> array_ids,
                            std::span<std::string_view const> array_values)
{
  auto get_type = [](std::string_view value) -> value_type {
    RTCX_EMBED_EXPECTS(!value.empty(), "Value cannot be empty");
    return std::isdigit(value[0]) ? value_type::INT : value_type::STRING;
  };

  using strings_t = std::vector<std::string_view>;
  using ints_t    = std::vector<std::int64_t>;
  using values_t  = std::variant<strings_t, ints_t>;

  std::map<std::string_view, values_t> arrays;

  for (size_t i = 0; i < array_ids.size(); ++i) {
    auto id    = array_ids[i];
    auto value = array_values[i];
    auto type  = get_type(value);
    if (auto array_it = arrays.find(id); array_it == arrays.end()) {
      switch (type) {
        case value_type::INT: {
          arrays.emplace(id, ints_t{});
        } break;
        case value_type::STRING: {
          arrays.emplace(id, strings_t{});
        } break;
        default: throw std::logic_error("Unexpected constant type");
      }
    }

    auto& array = arrays[id];

    switch (type) {
      case value_type::INT: {
        std::int64_t int_value;
        RTCX_EMBED_EXPECTS(
          std::from_chars(value.data(), value.data() + value.size(), int_value).ec == std::errc(),
          std::format("Invalid integer constant value: {}", value));
        std::get<ints_t>(array).push_back(int_value);
      } break;
      case value_type::STRING: {
        std::get<strings_t>(array).push_back(value);
      } break;

      default: break;
    }
  }

  std::string result;

  for (auto& [id, array] : arrays) {
    if (auto* ints = std::get_if<ints_t>(&array); ints != nullptr) {
      result +=
        std::format("constexpr std::int64_t {}[{}] = {{ {} }};\n\n",
                    id,
                    ints->size(),
                    join_formatted(*ints, ", ", [](std::int64_t v) { return std::to_string(v); }));
    } else {
      auto& strings = std::get<strings_t>(array);
      result +=
        std::format("constexpr char const* {}[{}] = {{ {} }};\n\n",
                    id,
                    strings.size(),
                    join_formatted(strings, ", ", [](auto s) { return std::format("\"{}\"", s); }));
    }
  }

  return result;
}

embed_output generate_cxx_source_files_data(std::string_view id,
                                            std::span<std::string_view const> array_ids,
                                            std::span<std::string_view const> array_values,
                                            std::span<std::string_view const> file_ids,
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

  auto [uncompressed_files_bytes, file_ranges] = merge_bytes_with_null_terminators(file_bytes);

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

  auto include_dirs_list =
    join_formatted(include_dirs, ",\n", [](auto s) { return std::format("\"{}\"", s); });
  auto file_dests_list =
    join_formatted(file_dsts, ",\n", [](auto s) { return std::format("\"{}\"", s); });
  auto file_ids_list =
    join_formatted(file_ids, ",\n", [](auto s) { return std::format("\"{}\"", s); });
  std::vector<size_t> file_indices(file_ids.size());
  std::iota(file_indices.begin(), file_indices.end(), 0ULL);
  auto file_indices_list = join_formatted(file_indices, "\n", [&](auto i) {
    return std::format("constexpr std::size_t {} = {}ULL;", file_ids[i], i);
  });
  auto file_ranges_list  = join_formatted(
    file_ranges, ",\n", [](auto r) { return std::format("{{{}, {}}}", r.offset, r.size); });
  auto hash_list = join_formatted(
    hash, ", ", [](uint8_t byte) { return std::format("0x{:02x}", static_cast<unsigned>(byte)); });
  auto arrays_list    = generate_arrays(array_ids, array_values);
  auto namespace_decl = "namespace " + std::string(id);

  auto cxx_header = std::format(
    R"***(
// Auto-generated header for embedded files
#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <string_view>

{} {{


constexpr char const * include_directories[{}] =
{{
{}
}};

{}

constexpr char const * file_ids[{}] =
{{
{}
}};

constexpr char const * file_destinations[{}] =
{{
{}
}};

constexpr std::size_t file_ranges[{}][2] =
{{
{}
}};

constexpr std::size_t files_uncompressed_size = {};

constexpr char const * files_compression = "{}";

extern "C" std::uint8_t const {}_files_begin[];

static std::span<std::uint8_t const> const files =
{{
{}_files_begin,
{}L
}};

constexpr std::uint8_t hash[{}] =
{{
{}
}};

{}

}}
)***",
    namespace_decl,
    include_dirs.size(),
    include_dirs_list,
    file_indices_list,
    file_ids.size(),
    file_ids_list,
    file_dsts.size(),
    file_dests_list,
    file_ranges.size(),
    file_ranges_list,
    uncompressed_files_bytes.size(),
    compression,
    id,
    id,
    binary_size,
    hash.size(),
    hash_list,
    arrays_list);

  auto asm_source = std::format(
    R"***(
.section .rodata
.global {0}_files_begin
{0}_files_begin:
.incbin "{0}.bin"

.section .note.GNU-stack,"",@progbits
)***",
    id);

  return embed_output{
    .cxx_header    = cxx_header,
    .asm_source    = asm_source,
    .bin_file_data = compress ? compressed_files_bytes : uncompressed_files_bytes};
}

void generate_embed(std::string_view id,
                    std::span<std::string_view const> array_ids,
                    std::span<std::string_view const> array_values,
                    std::span<std::string_view const> file_ids,
                    std::span<std::string_view const> file_paths,
                    std::span<std::string_view const> file_dsts,
                    std::span<std::string_view const> include_dirs,
                    std::string_view compression,
                    std::string_view output_directory)
{
  auto output = generate_cxx_source_files_data(
    id, array_ids, array_values, file_ids, file_paths, file_dsts, include_dirs, compression);

  std::filesystem::create_directories(std::filesystem::path(output_directory));

  std::ofstream header_file(std::format("{}/{}.hpp", output_directory, id));
  header_file << output.cxx_header;

  std::ofstream asm_file(std::format("{}/{}.s", output_directory, id));
  asm_file << output.asm_source;

  std::ofstream bin_file(std::format("{}/{}.bin", output_directory, id), std::ios::binary);
  bin_file.write(reinterpret_cast<char const*>(output.bin_file_data.data()),
                 static_cast<std::streamsize>(output.bin_file_data.size()));
}

std::vector<std::string_view> split_string(std::string_view str, char delimiter)
{
  if (str.empty()) { return {}; }
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
