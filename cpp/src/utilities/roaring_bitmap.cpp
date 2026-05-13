/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>

namespace cudf::iceberg {

namespace {

/// Roaring portable-format constants
constexpr uint32_t no_run_cookie             = 12'346;
constexpr uint32_t run_cookie                = 12'347;
constexpr uint32_t cookie_mask               = 0xFFFF;
constexpr uint32_t no_offset_threshold       = 4;
constexpr uint32_t max_array_container_card  = 4'096;
constexpr std::size_t bitset_container_bytes = 8'192;

constexpr std::size_t cookie_size         = sizeof(uint32_t);
constexpr std::size_t no_run_hdr_prefix   = 2 * sizeof(uint32_t);  // cookie + num_containers
constexpr std::size_t key_card_desc_size  = 2 * sizeof(uint16_t);  // key + card_minus_1
constexpr std::size_t offset_entry_size   = sizeof(uint32_t);
constexpr std::size_t run_descriptor_size = 2 * sizeof(uint16_t);  // start + length_minus_1

/**
 * @brief Loads a fixed width value from a string view without assuming aligned memory.
 */
template <typename T>
[[nodiscard]] T unaligned_load(std::string_view payload, std::size_t offset = 0)
{
  static_assert(cudf::is_fixed_width<T>(), "T must be a fixed width type");
  CUDF_EXPECTS(payload.size() >= offset + sizeof(T),
               "Roaring bitmap payload is too small to load field");
  T value;
  std::memcpy(&value, payload.data() + offset, sizeof(T));
  return value;
}

/**
 * @brief Parses the first bytes of a 32-bit roaring bitmap in portable format to extract the
 * cookie and the number of containers it encodes.
 *
 * - No-run (cookie == 12346): the next 4 bytes carry `num_containers`.
 * - Run (cookie & 0xFFFF == 12347): `num_containers - 1` is packed into the upper 16 bits of the
 *   cookie itself.
 */
[[nodiscard]] auto parse_cookie(std::string_view payload)
{
  auto const cookie = unaligned_load<uint32_t>(payload);
  if (cookie == no_run_cookie) {
    return std::pair{cookie, unaligned_load<uint32_t>(payload, cookie_size)};
  }
  if ((cookie & cookie_mask) == run_cookie) { return std::pair{cookie, (cookie >> 16) + 1}; }
  CUDF_FAIL("Invalid 32-bit roaring bitmap cookie: " + std::to_string(cookie));
}

/**
 * @brief Returns the cardinality of the specified container by reading the key-card descriptor at
 * `key_cards_offset + i * key_card_desc_size + sizeof(uint16_t)`.
 */
[[nodiscard]] auto container_cardinality(std::string_view payload,
                                         std::size_t key_cards_offset,
                                         uint32_t i)
{
  auto const card_minus_1 =
    unaligned_load<uint16_t>(payload, key_cards_offset + i * key_card_desc_size + sizeof(uint16_t));
  return static_cast<uint32_t>(card_minus_1) + 1;
}

/**
 * @brief Computes the block size of a no-run 32-bit roaring bitmap whose offset table has been
 * stripped (the only non-compliant case).
 */
[[nodiscard]] auto no_run_bitmap_stripped_block_size(std::string_view payload,
                                                     uint32_t num_containers)
{
  return std::accumulate(
    cuda::counting_iterator<uint32_t>(0),
    cuda::counting_iterator<uint32_t>(num_containers),
    std::size_t{no_run_hdr_prefix + num_containers * key_card_desc_size},
    [&](std::size_t acc, uint32_t container) {
      auto const card = container_cardinality(payload, no_run_hdr_prefix, container);
      return acc + ((card <= max_array_container_card) ? card * sizeof(uint16_t)
                                                       : bitset_container_bytes);
    });
}

/**
 * @brief Returns whether all containers in a no-run 32-bit roaring bitmap have
 * valid offsets
 */
[[nodiscard]] bool no_run_bitmap_has_valid_offsets(std::string_view payload,
                                                   uint32_t num_containers)
{
  if (num_containers == 0 or payload.empty()) { return true; }

  std::size_t const header_end       = no_run_hdr_prefix + num_containers * key_card_desc_size;
  std::size_t const offset_table_end = header_end + num_containers * offset_entry_size;
  if (payload.size() < offset_table_end) { return false; }

  auto expected = static_cast<uint32_t>(offset_table_end);
  return std::all_of(
    cuda::counting_iterator<uint32_t>(0),
    cuda::counting_iterator<uint32_t>(num_containers),
    [&](uint32_t i) {
      auto const stored = unaligned_load<uint32_t>(payload, header_end + i * offset_entry_size);
      if (stored != expected) { return false; }
      auto const card = container_cardinality(payload, no_run_hdr_prefix, i);
      expected += static_cast<uint32_t>(
        (card <= max_array_container_card) ? card * sizeof(uint16_t) : bitset_container_bytes);
      return expected <= payload.size();
    });
}

/**
 * @brief Computes the block size of a spec-compliant 32-bit roaring bitmap by walking every
 * container
 *
 * The walk inspects each container's type (array, bitset, or run-encoded) and accumulates its body
 * size. Run-encoded containers (only present under the run cookie) embed their size as a leading
 * `num_runs` uint16 followed by `num_runs` (start, length_minus1) pairs; array containers use
 * `cardinality * 2` bytes; bitset containers use a fixed 8192 bytes.
 */
[[nodiscard]] auto compliant_block_size(std::string_view payload)
{
  auto const [cookie, num_containers] = parse_cookie(payload);
  if (cookie == no_run_cookie and num_containers == 0) { return no_run_hdr_prefix; }

  auto const has_run                  = (cookie & cookie_mask) == run_cookie;
  std::size_t const run_bitmap_offset = cookie_size;
  auto const run_bitmap_size =
    has_run ? cudf::util::div_rounding_up_safe<std::size_t>(num_containers, 8) : std::size_t{0};
  std::size_t const key_cards_offset =
    (has_run ? cookie_size : no_run_hdr_prefix) + run_bitmap_size;
  std::size_t const header_end = key_cards_offset + num_containers * key_card_desc_size;
  bool const offsets_present   = (not has_run) or num_containers >= no_offset_threshold;
  std::size_t const containers_start =
    header_end + (offsets_present ? num_containers * offset_entry_size : std::size_t{0});

  return std::accumulate(
    cuda::counting_iterator<uint32_t>(0),
    cuda::counting_iterator<uint32_t>(num_containers),
    containers_start,
    [&](std::size_t pos, uint32_t i) {
      bool const is_run_container =
        has_run and ((static_cast<uint8_t>(payload[run_bitmap_offset + i / 8]) >> (i % 8)) & 1u);
      if (is_run_container) {
        auto const num_runs = unaligned_load<uint16_t>(payload, pos);
        return pos + sizeof(uint16_t) + num_runs * run_descriptor_size;
      }
      auto const card = container_cardinality(payload, key_cards_offset, i);
      return pos + ((card <= max_array_container_card) ? card * sizeof(uint16_t)
                                                       : bitset_container_bytes);
    });
}

/**
 * @brief Injects the missing offset table into a no-run 32-bit roaring bitmap with at least one
 * container. The offset table is placed between the key-card descriptors and the container data.
 */
[[nodiscard]] std::string inject_no_run_offsets(std::string_view payload,
                                                uint32_t num_containers,
                                                std::size_t block_size)
{
  std::size_t const header_end       = no_run_hdr_prefix + num_containers * key_card_desc_size;
  std::size_t const offset_table_len = num_containers * offset_entry_size;

  std::string compliant_payload;
  compliant_payload.reserve(block_size + offset_table_len);
  compliant_payload.append(payload.data(), header_end);

  // Compute cumulative offsets; base starts right after the injected offset table.
  std::ignore = std::accumulate(
    cuda::counting_iterator<uint32_t>(0),
    cuda::counting_iterator<uint32_t>(num_containers),
    static_cast<uint32_t>(header_end + offset_table_len),
    [&](uint32_t base, uint32_t i) {
      compliant_payload.append(reinterpret_cast<char const*>(&base), offset_entry_size);
      auto const card = container_cardinality(payload, no_run_hdr_prefix, i);
      return base + static_cast<uint32_t>((card <= max_array_container_card)
                                            ? card * sizeof(uint16_t)
                                            : bitset_container_bytes);
    });

  compliant_payload.append(payload.data() + header_end, block_size - header_end);
  return compliant_payload;
}

}  // namespace

/**
 * @copydoc cudf::iceberg::is_roaring_bitmap_compliant
 */
bool is_roaring_bitmap_compliant(roaring_bitmap_type type, std::string_view payload)
{
  if (type == roaring_bitmap_type::BITS_32) {
    auto const [cookie, num_containers] = parse_cookie(payload);
    return not(cookie == no_run_cookie and num_containers > 0 and
               not no_run_bitmap_has_valid_offsets(payload, num_containers));
  } else {
    auto constexpr bucket_key_field_size = sizeof(uint32_t);

    auto const num_keys = unaligned_load<uint64_t>(payload);
    if (num_keys == 0) { return true; }

    std::size_t data_pos = sizeof(uint64_t);

    return std::all_of(
      cuda::counting_iterator<uint64_t>(0), cuda::counting_iterator<uint64_t>(num_keys), [&](auto) {
        CUDF_EXPECTS(data_pos + bucket_key_field_size <= payload.size(),
                     "Roaring bitmap payload is too small to contain the bucket key");
        data_pos += bucket_key_field_size;

        auto const bucket = std::string_view(payload).substr(data_pos);
        if (not is_roaring_bitmap_compliant(roaring_bitmap_type::BITS_32, bucket)) { return false; }
        data_pos += compliant_block_size(bucket);
        return true;
      });
  }
}

/**
 * @copydoc cudf::iceberg::make_compliant_roaring_bitmap
 */
std::string make_compliant_roaring_bitmap(roaring_bitmap_type type, std::string_view payload)
{
  if (type == roaring_bitmap_type::BITS_32) {
    if (is_roaring_bitmap_compliant(roaring_bitmap_type::BITS_32, payload)) {
      return std::string{payload};
    }
    auto const [_, num_containers] = parse_cookie(payload);
    auto const block_size          = no_run_bitmap_stripped_block_size(payload, num_containers);
    return inject_no_run_offsets(payload.substr(0, block_size), num_containers, block_size);
  } else {
    auto const num_keys = unaligned_load<uint64_t>(payload);

    std::string out;
    // Generous reserve: an injected offset table grows a bucket by at most `num_containers * 4`
    // bytes, which is always smaller than the bucket's own container body. Doubling is plenty.
    out.reserve(payload.size() * 2);
    out.append(payload.data(), sizeof(uint64_t));

    std::size_t pos = sizeof(uint64_t);
    std::for_each(
      cuda::counting_iterator<uint64_t>(0), cuda::counting_iterator<uint64_t>(num_keys), [&](auto) {
        CUDF_EXPECTS(pos + sizeof(uint32_t) <= payload.size(),
                     "Roaring bitmap payload is too small to contain the bucket key");
        out.append(payload.data() + pos, sizeof(uint32_t));
        pos += sizeof(uint32_t);

        auto const bucket = std::string_view(payload).substr(pos);
        if (not is_roaring_bitmap_compliant(roaring_bitmap_type::BITS_32, bucket)) {
          auto const [_, num_containers] = parse_cookie(bucket);
          auto const block_size = no_run_bitmap_stripped_block_size(bucket, num_containers);
          out.append(
            inject_no_run_offsets(bucket.substr(0, block_size), num_containers, block_size));
          pos += block_size;
        } else {
          auto const block_size = compliant_block_size(bucket);
          out.append(bucket.data(), block_size);
          pos += block_size;
        }
      });

    return out;
  }
}

}  // namespace cudf::iceberg
