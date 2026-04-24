/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <numeric>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudf::iceberg {

namespace {

/**
 * @brief Loads a fixed width value from a string view without assuming aligned
 * memory
 */
template <typename T>
  requires(std::is_integral_v<T>)
T unaligned_load(std::string_view payload, std::size_t offset = 0)
{
  CUDF_EXPECTS(
    payload.size() >= offset + sizeof(T),
    "Roaring bitmap payload is too small to load field of type " + std::string(typeid(T).name()));
  T value;
  std::memcpy(&value, payload.data() + offset, sizeof(T));
  return value;
}

/// Constants for the 32-bit roaring bitmap cookies
constexpr uint32_t no_run_cookie  = 12'346;
constexpr uint32_t run_cookie     = 12'347;
constexpr uint32_t cookie_mask    = 0xFFFF;
constexpr std::size_t cookie_size = sizeof(uint32_t);

/**
 * @brief Parses the first 4 or 8 bytes of a 32-bit roaring bitmap in portable format to
 * extract the cookie and container count
 *
 * - No-run (cookie == 12346): followed by a 4-byte num_containers field.
 * - Run (cookie & 0xFFFF == 12347): upper 16 bits encode num_containers - 1.
 */
std::pair<uint32_t, uint32_t> parse_roaring32_cookie(std::string_view payload)
{
  auto const cookie = unaligned_load<uint32_t>(payload);

  if (cookie == no_run_cookie) {
    auto const num_containers = unaligned_load<uint32_t>(payload.substr(cookie_size));
    return {cookie, num_containers};
  } else if ((cookie & cookie_mask) == run_cookie) {
    return {cookie, (cookie >> 16) + 1};
  }

  CUDF_FAIL("Invalid 32-bit roaring bitmap cookie: " + std::to_string(cookie));
}

/// Constants for the 32-bit roaring bitmap
constexpr uint32_t no_offset_threshold      = 4;
constexpr uint32_t max_array_container_card = 4'096;
constexpr uint32_t bitset_container_bytes   = 8'192;

constexpr std::size_t container_count_size = sizeof(uint32_t);
constexpr std::size_t no_run_header_prefix = cookie_size + container_count_size;
constexpr std::size_t key_card_desc_size   = sizeof(uint16_t) + sizeof(uint16_t);
constexpr std::size_t offset_entry_size    = sizeof(uint32_t);

constexpr std::size_t run_pair_size = sizeof(uint16_t) + sizeof(uint16_t);
constexpr std::size_t num_runs_size = sizeof(uint16_t);

/**
 * @brief Computes the total serialized block size for a block with no-run cookie
 * by walking the key-card descriptors to sum up container data sizes
 *
 * Roaring32 portable layout (no-run variant):
 *   [cookie          4B]  (uint32 == 12346)
 *   [num_containers  4B]  (uint32)
 *   [key-card descriptors  num_containers * 4B]  (key:u16, card_minus1:u16)
 *   [offset table    num_containers * 4B]  (only if num_containers >= 4)
 *   [container data  variable]
 */
std::size_t no_run_block_size(std::string_view payload, uint32_t num_containers, bool has_offsets)
{
  std::size_t hdr = no_run_header_prefix + num_containers * key_card_desc_size;
  if (has_offsets) { hdr += num_containers * offset_entry_size; }

  return std::accumulate(
    std::views::iota(uint32_t{0}, num_containers).begin(),
    std::views::iota(uint32_t{0}, num_containers).end(),
    hdr,
    [&](std::size_t acc, uint32_t c) {
      auto const card_minus1 = unaligned_load<uint16_t>(
        payload, no_run_header_prefix + c * key_card_desc_size + sizeof(uint16_t));
      uint32_t const card = static_cast<uint32_t>(card_minus1) + 1;
      return acc + ((card <= max_array_container_card) ? card * sizeof(uint16_t)
                                                       : bitset_container_bytes);
    });
}

/**
 * @brief Computes the total serialized block size for a block with run cookie (12347)
 * by walking each container's run-length data
 *
 * Roaring32 portable layout (run variant):
 *   [cookie          4B]  (lower 16 bits == 12347, upper 16 = num_containers-1)
 *   [run bitmap      ceil(num_containers/8) B]
 *   [key-card descriptors  num_containers * 4B]
 *   [offset table    num_containers * 4B]  (only if num_containers >= 4)
 *   [run data        variable]  (num_runs:uint16, then num_runs * (start:uint16,
 *   len-1:uint16))
 */
std::size_t run_block_size(std::string_view payload, uint32_t num_containers)
{
  // Run bitmap follows the 4-byte cookie, one bit per container.
  std::size_t const run_bitmap_size = (num_containers + 7) / 8;
  std::size_t const kc_offset       = cookie_size + run_bitmap_size;
  bool const has_offsets            = (num_containers >= no_offset_threshold);
  std::size_t hdr                   = kc_offset + num_containers * key_card_desc_size;
  if (has_offsets) { hdr += num_containers * offset_entry_size; }

  return std::accumulate(std::views::iota(uint32_t{0}, num_containers).begin(),
                         std::views::iota(uint32_t{0}, num_containers).end(),
                         hdr,
                         [&](std::size_t pos, uint32_t) {
                           auto const num_runs = unaligned_load<uint16_t>(payload, pos);
                           return pos + num_runs_size + num_runs * run_pair_size;
                         });
}

/**
 * @brief Computes the total serialized block size for a 32-bit roaring bitmap
 */
std::size_t roaring32_block_size(std::string_view payload)
{
  auto const [cookie, num_containers] = parse_roaring32_cookie(payload);
  if ((cookie & cookie_mask) == run_cookie) { return run_block_size(payload, num_containers); }
  return no_run_block_size(payload, num_containers, num_containers >= no_offset_threshold);
}

/**
 * @brief Checks whether a 32-bit roaring bitmap payload is normalized for cudf::roaring_bitmap
 *
 *   - No-run cookie (12346), num_containers in [1,3]: offset table is omitted
 *     per the portable spec but cudf::roaring_bitmap requires it; must inject dummy offsets.
 *   - Run cookie (12347): cudf::roaring_bitmap only accepts the no-run portable format;
 *     must convert run-encoded containers to array/bitset containers.
 */
template <roaring_bitmap_type Type>
bool is_bitmap_normalized(std::string_view payload)
  requires(Type == roaring_bitmap_type::BITS_32)
{
  auto const [cookie, num_containers] = parse_roaring32_cookie(payload);
  if (cookie == no_run_cookie) {
    return not(num_containers > 0 && num_containers < no_offset_threshold);
  }
  return false;
}

/**
 * @brief Checks whether all embedded 32-bit roaring bitmaps in a 64-bit
 * roaring bitmap payload are normalized for cudf::roaring_bitmap
 */
template <roaring_bitmap_type Type>
bool is_bitmap_normalized(std::string_view payload)
  requires(Type == roaring_bitmap_type::BITS_64)
{
  auto const num_keys = unaligned_load<uint64_t>(payload);

  // Skip over the num_keys (8 bytes) prefix.
  std::size_t pos = sizeof(uint64_t);

  CUDF_EXPECTS(pos + sizeof(uint32_t) <= payload.size(),
               "Roaring bitmap payload is too small to contain the `key` field");

  return std::all_of(std::views::iota(uint64_t{0}, num_keys).begin(),
                     std::views::iota(uint64_t{0}, num_keys).end(),
                     [&](uint64_t) {
                       if (pos + sizeof(uint32_t) > payload.size()) { return true; }
                       // Skip over the key (4 bytes)
                       pos += sizeof(uint32_t);
                       // Get the 32-bit roaring bitmap
                       auto const payload32 = payload.substr(pos);
                       pos += roaring32_block_size(payload32);
                       return is_bitmap_normalized<roaring_bitmap_type::BITS_32>(payload32);
                     });
}

/**
 * @brief Injects the missing offset table into a no-run bitmap whose
 * num_containers < no_offset_threshold
 *
 * The offset table is placed between the key-card descriptors and the container data.
 */
std::string inject_no_run_offsets(std::string_view payload, uint32_t num_containers)
{
  std::size_t const header_end = no_run_header_prefix + num_containers * key_card_desc_size;
  std::size_t const offset_section_size = num_containers * offset_entry_size;
  std::string out;
  out.reserve(payload.size() + offset_section_size);
  out.append(payload.data(), header_end);

  // Compute cumulative offsets; base starts right after the injected offsets.
  std::accumulate(std::views::iota(uint32_t{0}, num_containers).begin(),
                  std::views::iota(uint32_t{0}, num_containers).end(),
                  static_cast<uint32_t>(header_end + offset_section_size),
                  [&](uint32_t base, uint32_t container) {
                    out.append(reinterpret_cast<char const*>(&base), offset_entry_size);
                    auto const card_minus1 = unaligned_load<uint16_t>(
                      payload,
                      no_run_header_prefix + container * key_card_desc_size + sizeof(uint16_t));
                    uint32_t const card = static_cast<uint32_t>(card_minus1) + 1;
                    return base + ((card <= max_array_container_card) ? card * sizeof(uint16_t)
                                                                      : bitset_container_bytes);
                  });

  out.append(payload.data() + header_end, payload.size() - header_end);
  return out;
}

/**
 * @brief Converts a run-encoded Roaring32 (cookie 12347, num_containers < 4) to
 * a no-run Roaring32 (cookie 12346) by expanding runs into sorted arrays
 *
 * The result always includes offset headers.
 *
 * cudf::roaring_bitmap's metadata parser rejects run bitmaps with < 4 containers outright
 * (the `contains_run_container` device code exists but the host parser
 * doesn't reach it), so we must convert to array format on the host.
 */
std::string convert_run_to_no_run(std::string_view payload, uint32_t num_containers)
{
  // In the run variant the key-card descriptors start after the cookie and
  // the per-container run bitmap.
  std::size_t const run_bitmap_size = (num_containers + 7) / 8;
  std::size_t const kc_offset       = cookie_size + run_bitmap_size;

  struct container_info {
    uint16_t key;
    std::vector<uint16_t> expanded_values;
  };
  std::vector<container_info> containers(num_containers);

  // Run data follows the key-card descriptors (and offsets, if present).
  std::size_t data_pos = kc_offset + num_containers * key_card_desc_size;
  std::ranges::for_each(std::views::iota(uint32_t{0}, num_containers), [&](uint32_t c) {
    containers[c].key   = unaligned_load<uint16_t>(payload, kc_offset + c * key_card_desc_size);
    auto const num_runs = unaligned_load<uint16_t>(payload, data_pos);
    data_pos += num_runs_size;
    // For each run, extract the start and length and add the values to the
    // container
    auto& vals = containers[c].expanded_values;
    std::ranges::for_each(std::views::iota(uint16_t{0}, num_runs), [&](uint16_t) {
      auto const start  = static_cast<uint32_t>(unaligned_load<uint16_t>(payload, data_pos));
      auto const length = unaligned_load<uint16_t>(payload, data_pos + sizeof(uint16_t)) + 1;
      data_pos += run_pair_size;
      std::ranges::transform(std::views::iota(start, start + length),
                             std::back_inserter(vals),
                             [](uint32_t v) { return static_cast<uint16_t>(v); });
    });
  });

  // Emit a no-run portable block with offsets always included.
  std::string out;
  auto const cookie = no_run_cookie;
  out.append(reinterpret_cast<char const*>(&cookie), cookie_size);
  out.append(reinterpret_cast<char const*>(&num_containers), container_count_size);

  for (auto const& ci : containers) {
    auto const card_minus1 = static_cast<uint16_t>(ci.expanded_values.size() - 1);
    out.append(reinterpret_cast<char const*>(&ci.key), sizeof(uint16_t));
    out.append(reinterpret_cast<char const*>(&card_minus1), sizeof(uint16_t));
  }

  // Offset table: each entry points to the start of that container's data.
  std::ignore = std::accumulate(
    containers.begin(),
    containers.end(),
    static_cast<uint32_t>(no_run_header_prefix + num_containers * key_card_desc_size +
                          num_containers * offset_entry_size),
    [&](uint32_t base, auto const& ci) {
      out.append(reinterpret_cast<char const*>(&base), offset_entry_size);
      return base + static_cast<uint32_t>(ci.expanded_values.size()) * sizeof(uint16_t);
    });

  for (auto const& ci : containers) {
    for (auto const v : ci.expanded_values) {
      out.append(reinterpret_cast<char const*>(&v), sizeof(uint16_t));
    }
  }

  return out;
}

/**
 * @brief Normalizes a single 32-bit roaring bitmap for cudf::roaring_bitmap
 */
template <roaring_bitmap_type Type>
std::string normalize_roaring(std::string_view payload)
  requires(Type == roaring_bitmap_type::BITS_32)
{
  auto const [cookie, num_containers] = parse_roaring32_cookie(payload);
  if ((cookie & cookie_mask) == run_cookie) {
    return convert_run_to_no_run(payload, num_containers);
  }
  std::size_t const block_size = no_run_block_size(payload, num_containers, false);
  return inject_no_run_offsets(payload.substr(0, block_size), num_containers);
}

/**
 * @brief Walks the 64-bit roaring bitmap payload and normalizes each 32-bit
 * roaring bitmap bucket for cudf::roaring_bitmap
 */
template <roaring_bitmap_type Type>
std::string normalize_roaring(std::string_view payload)
  requires(Type == roaring_bitmap_type::BITS_64)
{
  auto const num_keys = unaligned_load<uint64_t>(payload);

  std::string normalized;
  normalized.reserve(payload.size() + num_keys * 16);
  normalized.append(payload.data(), sizeof(uint64_t));
  std::size_t pos = sizeof(uint64_t);

  std::ranges::for_each(std::views::iota(uint64_t{0}, num_keys), [&](uint64_t) {
    // Skip over the key (4 bytes)
    if (pos + sizeof(uint32_t) > payload.size()) { return; }
    // Append the cookie
    normalized.append(payload.data() + pos, cookie_size);
    pos += cookie_size;

    // TODO(mh): Append the remaining data if less than 4 bytes are left.
    // Should we throw here instead?
    if (pos + sizeof(uint32_t) > payload.size()) {
      normalized.append(payload.data() + pos, payload.size() - pos);
      return;
    }

    auto const payload32  = payload.substr(pos);
    auto const block_size = roaring32_block_size(payload32);
    // Append the 32-bit roaring bitmap
    if (is_bitmap_normalized<roaring_bitmap_type::BITS_32>(payload32)) {
      normalized.append(payload32.data(), block_size);
    } else {
      normalized.append(
        normalize_roaring<roaring_bitmap_type::BITS_32>(payload32.substr(0, block_size)));
    }
    pos += block_size;
  });

  return normalized;
}

}  // namespace

/**
 * @copydoc cudf::iceberg::is_puffin_payload_normalized
 */
bool is_roaring_bitmap_normalized(roaring_bitmap_type type, std::string_view payload)
{
  if (type == roaring_bitmap_type::BITS_32) {
    return is_bitmap_normalized<roaring_bitmap_type::BITS_32>(payload);
  }
  return is_bitmap_normalized<roaring_bitmap_type::BITS_64>(payload);
}

/**
 * @copydoc cudf::iceberg::normalize_puffin_payload
 */
std::string normalize_roaring_bitmap(roaring_bitmap_type type, std::string_view payload)
{
  if (type == roaring_bitmap_type::BITS_32) {
    return normalize_roaring<roaring_bitmap_type::BITS_32>(payload);
  }
  return normalize_roaring<roaring_bitmap_type::BITS_64>(payload);
}

}  // namespace cudf::iceberg
