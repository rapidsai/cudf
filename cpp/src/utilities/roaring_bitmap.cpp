/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>

namespace cudf::iceberg {

namespace {

/**
 * @brief Loads a fixed width value from a string view without assuming aligned
 * memory
 */
template <typename T>
  requires(std::is_integral_v<T>)
[[nodiscard]] T unaligned_load(std::string_view payload, std::size_t offset = 0)
{
  CUDF_EXPECTS(
    payload.size() >= offset + sizeof(T),
    "Roaring bitmap payload is too small to load field of type " + std::string(typeid(T).name()));
  T value;
  std::memcpy(&value, payload.data() + offset, sizeof(T));
  return value;
}

/// Roaring bitmap cookie related constants
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

/**
 * @brief Checks whether a container is a run container
 *
 * @param payload The payload of the 32-bit roaring bitmap
 * @param container The index of the container
 * @return Whether the container is a run container
 */
[[nodiscard]] inline bool is_run_container(std::string_view payload, uint32_t container)
{
  auto const run_bitmap_byte = unaligned_load<uint8_t>(payload, cookie_size + container / CHAR_BIT);
  return (run_bitmap_byte & (1 << (container % CHAR_BIT))) != 0;
}

/// 32-bit roaring bitmap related constants
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
 * @brief Checks whether a no-run bitmap contains a valid offset table
 */
[[nodiscard]] bool no_run_has_valid_offsets(std::string_view payload, uint32_t num_containers)
{
  std::size_t const header_end = no_run_header_prefix + num_containers * key_card_desc_size;
  if (payload.size() < header_end + num_containers * offset_entry_size) { return false; }

  auto expected_offset = static_cast<size_t>(header_end + num_containers * offset_entry_size);
  return std::all_of(
    cuda::counting_iterator<uint32_t>(0),
    cuda::counting_iterator(num_containers),
    [&](auto container) mutable {
      auto const offset =
        unaligned_load<uint32_t>(payload, header_end + container * offset_entry_size);
      if (offset != expected_offset) { return false; }

      // Update the expected offset for the next container
      auto const card_minus1 = unaligned_load<uint16_t>(
        payload, no_run_header_prefix + container * key_card_desc_size + sizeof(uint16_t));
      auto const card = static_cast<uint32_t>(card_minus1) + 1;
      expected_offset +=
        (card <= max_array_container_card) ? card * sizeof(uint16_t) : bitset_container_bytes;

      // Verify the container's bytes fit within the payload
      if (expected_offset > payload.size()) { return false; }

      return true;
    });
}

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
[[nodiscard]] std::size_t no_run_block_size(std::string_view payload,
                                            uint32_t num_containers,
                                            bool has_offsets)
{
  std::size_t hdr = no_run_header_prefix + num_containers * key_card_desc_size;
  if (has_offsets) { hdr += num_containers * offset_entry_size; }

  return std::accumulate(
    cuda::counting_iterator<uint32_t>(0),
    cuda::counting_iterator(num_containers),
    hdr,
    [&](std::size_t data_pos, auto container) {
      auto const card_minus1 = unaligned_load<uint16_t>(
        payload, no_run_header_prefix + container * key_card_desc_size + sizeof(uint16_t));
      auto const card = static_cast<uint32_t>(card_minus1) + 1;
      return data_pos + ((card <= max_array_container_card) ? card * sizeof(uint16_t)
                                                            : bitset_container_bytes);
    });
}

/**
 * @brief Computes the total serialized block size for a block with run cookie (12347)
 * by walking each container's serialized data
 *
 * Roaring32 portable layout (run variant):
 *   [cookie          4B]  (lower 16 bits == 12347, upper 16 = num_containers-1)
 *   [run bitmap      ceil(num_containers/8) B]
 *   [key-card descriptors  num_containers * 4B]
 *   [offset table    num_containers * 4B]  (only if num_containers >= 4)
 *   [container data  variable]  (run, array, or bitset depending on the run bitmap and cardinality)
 */
[[nodiscard]] std::size_t run_block_size(std::string_view payload, uint32_t num_containers)
{
  // Run bitmap follows the 4-byte cookie, one bit per container.
  std::size_t const run_bitmap_size = (num_containers + 7) / 8;
  std::size_t const kc_offset       = cookie_size + run_bitmap_size;
  bool const has_offsets            = (num_containers >= no_offset_threshold);
  std::size_t hdr                   = kc_offset + num_containers * key_card_desc_size;
  if (has_offsets) { hdr += num_containers * offset_entry_size; }

  return std::accumulate(
    cuda::counting_iterator<uint32_t>(0),
    cuda::counting_iterator(num_containers),
    hdr,
    [&](std::size_t data_pos, auto container) {
      auto const card_minus1 = unaligned_load<uint16_t>(
        payload, kc_offset + container * key_card_desc_size + sizeof(uint16_t));
      auto const card = static_cast<uint32_t>(card_minus1) + 1;
      if (is_run_container(payload, container)) {
        auto const num_runs = unaligned_load<uint16_t>(payload, data_pos);
        return data_pos + num_runs_size + num_runs * run_pair_size;
      }
      return data_pos + ((card <= max_array_container_card) ? card * sizeof(uint16_t)
                                                            : bitset_container_bytes);
    });
}

/**
 * @brief Computes the total serialized block size for a 32-bit roaring bitmap
 */
[[nodiscard]] std::size_t roaring32_block_size(std::string_view payload)
{
  // Dispatch block size computation based on the cookie type
  auto const [cookie, num_containers] = parse_roaring32_cookie(payload);
  if ((cookie & cookie_mask) == run_cookie) { return run_block_size(payload, num_containers); }
  return no_run_block_size(
    payload, num_containers, no_run_has_valid_offsets(payload, num_containers));
}

/**
 * @brief Checks whether a 32-bit roaring bitmap payload is normalized for cudf::roaring_bitmap
 */
template <roaring_bitmap_type Type>
[[nodiscard]] inline bool is_bitmap_normalized(std::string_view payload)
  requires(Type == roaring_bitmap_type::BITS_32)
{
  auto const [cookie, num_containers] = parse_roaring32_cookie(payload);

  // 32-bit roaring bitmap is normalized if it has a no-run cookie and a valid offset table
  return (cookie == no_run_cookie) and no_run_has_valid_offsets(payload, num_containers);
}

/**
 * @brief Checks whether all embedded 32-bit roaring bitmaps in a 64-bit
 * roaring bitmap payload are normalized for cudf::roaring_bitmap
 */
template <roaring_bitmap_type Type>
[[nodiscard]] inline bool is_bitmap_normalized(std::string_view payload)
  requires(Type == roaring_bitmap_type::BITS_64)
{
  auto const num_keys = unaligned_load<uint64_t>(payload);

  // No keys, bitmap is normalized
  if (num_keys == 0) { return true; }

  // Skip over the num_keys (8 bytes) field
  std::size_t pos = sizeof(uint64_t);

  CUDF_EXPECTS(pos + sizeof(uint32_t) <= payload.size(),
               "Roaring bitmap payload is too small to contain the `key` field");

  return std::all_of(
    cuda::counting_iterator<uint64_t>(0), cuda::counting_iterator<uint64_t>(num_keys), [&](auto) {
      // A missing high-key indicates a truncated payload, reject it.
      if (pos + sizeof(uint32_t) > payload.size()) { return false; }
      // Skip over the key (4 bytes)
      pos += sizeof(uint32_t);
      // Get the 32-bit roaring bitmap
      auto const payload32 = payload.substr(pos);
      pos += roaring32_block_size(payload32);
      return is_bitmap_normalized<roaring_bitmap_type::BITS_32>(payload32);
    });
}

}  // namespace

/**
 * @copydoc cudf::iceberg::is_puffin_payload_normalized
 */
CUDF_EXPORT bool is_roaring_bitmap_normalized(roaring_bitmap_type type, std::string_view payload)
{
  // Dispatch based on the roaring bitmap type
  if (type == roaring_bitmap_type::BITS_32) {
    return is_bitmap_normalized<roaring_bitmap_type::BITS_32>(payload);
  }
  return is_bitmap_normalized<roaring_bitmap_type::BITS_64>(payload);
}

}  // namespace cudf::iceberg
