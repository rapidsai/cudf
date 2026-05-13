/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "roaring_bitmap_utils.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <cstring>
#include <ranges>
#include <string_view>
#include <unordered_set>
#include <vector>

template <typename T>
struct RoaringBitmapTest : public cudf::test::BaseFixture {};

using RoaringTypes = cudf::test::Types<cuda::std::uint32_t, cuda::std::uint64_t>;

TYPED_TEST_SUITE(RoaringBitmapTest, RoaringTypes);

TYPED_TEST(RoaringBitmapTest, Basics)
{
  auto constexpr num_keys = 100'000;
  using Key               = TypeParam;

  auto insert_keys = std::vector<Key>(num_keys / 2);
  std::generate(insert_keys.begin(), insert_keys.end(), [k = Key{0}]() mutable {
    auto const result = k;
    k += 2;
    return result;
  });

  auto const [serialized_bitmap_data, bitmap_type, col_type] = [&]() {
    if constexpr (std::is_same_v<Key, cuda::std::uint64_t>) {
      return std::make_tuple(serialize_roaring_bitmap<Key>(insert_keys),
                             cudf::roaring_bitmap_type::BITS_64,
                             cudf::type_id::UINT64);
    } else {
      return std::make_tuple(serialize_roaring_bitmap<Key>(insert_keys),
                             cudf::roaring_bitmap_type::BITS_32,
                             cudf::type_id::UINT32);
    }
  }();

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto bitmap = cudf::roaring_bitmap(bitmap_type, serialized_bitmap_data);

  auto key_iter = cuda::counting_iterator<Key>(0);
  auto keys_col = cudf::test::fixed_width_column_wrapper<Key>(key_iter, key_iter + num_keys);

  auto const is_even =
    cudf::detail::make_counting_transform_iterator(0, [](auto const i) { return i % 2 == 0; });

  {
    auto result_col = bitmap.contains_async(keys_col, stream, mr);
    auto results    = cudf::detail::make_host_vector_async(
      cudf::device_span<bool const>(result_col->view().template data<bool>(), num_keys), stream);
    stream.synchronize();
    EXPECT_TRUE(std::equal(results.begin(), results.end(), is_even));
  }

  {
    auto result_iter = cuda::constant_iterator<bool>(false);
    auto result_col =
      cudf::test::fixed_width_column_wrapper<bool>(result_iter, result_iter + num_keys).release();
    bitmap.contains_async(keys_col, result_col->mutable_view(), stream);
    auto results = cudf::detail::make_host_vector_async(
      cudf::device_span<bool const>(result_col->view().template data<bool>(), num_keys), stream);
    stream.synchronize();
    EXPECT_TRUE(std::equal(results.begin(), results.end(), is_even));
  }
}
struct RoaringBitmapErrorTest : public cudf::test::BaseFixture {};

TEST_F(RoaringBitmapErrorTest, EmptySerializedBitmapData)
{
  EXPECT_THROW(auto bitmap = cudf::roaring_bitmap(cudf::roaring_bitmap_type::BITS_64, {}),
               std::invalid_argument);
}

TEST_F(RoaringBitmapErrorTest, TypeMismatch)
{
  auto insert_keys            = std::vector<cuda::std::uint64_t>{1, 2, 3};
  auto serialized_bitmap_data = serialize_roaring_bitmap<cuda::std::uint64_t>(insert_keys);
  auto const stream           = cudf::get_default_stream();
  auto bitmap = cudf::roaring_bitmap(cudf::roaring_bitmap_type::BITS_64, serialized_bitmap_data);
  bitmap.materialize(stream);
  auto probe_keys = cudf::test::fixed_width_column_wrapper<cuda::std::uint32_t>{1, 2, 3}.release();
  EXPECT_THROW(std::ignore = bitmap.contains_async(
                 probe_keys->view(), stream, cudf::get_current_device_resource_ref()),
               std::invalid_argument);
}

TEST_F(RoaringBitmapErrorTest, EmptyProbeKeys)
{
  auto insert_keys            = std::vector<cuda::std::uint64_t>{1, 2, 3};
  auto serialized_bitmap_data = serialize_roaring_bitmap<cuda::std::uint64_t>(insert_keys);
  auto const stream           = cudf::get_default_stream();
  auto bitmap = cudf::roaring_bitmap(cudf::roaring_bitmap_type::BITS_64, serialized_bitmap_data);
  auto probe_keys = cudf::make_empty_column(cudf::data_type{cudf::type_id::UINT64});
  EXPECT_NO_THROW(std::ignore = bitmap.contains_async(
                    probe_keys->view(), stream, cudf::get_current_device_resource_ref()));
}

namespace {

/**
 * @brief Maps a `Key` type to its corresponding `roaring_bitmap_type`
 */
template <typename Key>
constexpr auto bitmap_type_v =
  std::is_same_v<Key, cuda::std::uint64_t> ? cudf::roaring_bitmap_type::BITS_64
                                           : cudf::roaring_bitmap_type::BITS_32;

/**
 * @brief Composes a 32-bit or 64-bit key from a high-32-bit and low-32-bit half
 */
template <typename Key>
Key make_key(uint32_t high_bits, uint32_t low_bits)
{
  if constexpr (std::is_same_v<Key, cuda::std::uint64_t>) {
    return (static_cast<Key>(high_bits) << 32) | low_bits;
  } else {
    return low_bits;
  }
}

/**
 * @brief Builds a vector of `Key`s by pairing `high_bits` with each value in a range of low bits
 */
template <typename Key>
auto make_keys(uint32_t high_bits, std::ranges::input_range auto&& low_bits)
{
  auto keys = std::vector<Key>{};
  keys.reserve(std::ranges::distance(low_bits));
  std::ranges::transform(low_bits, std::back_inserter(keys), [high_bits](uint32_t lo) {
    return make_key<Key>(high_bits, lo);
  });
  return keys;
}

/**
 * @brief Generates keys that span multiple roaring containers within a single 32-bit block
 */
template <typename Key>
std::vector<Key> make_many_container_keys(uint32_t high_bits       = 0,
                                          uint32_t keys_per_bucket = 100,
                                          uint32_t num_buckets     = 4)
{
  auto low_bits = std::views::iota(uint32_t{0}, num_buckets * keys_per_bucket) |
                  std::views::transform([&](auto i) {
                    return ((i / keys_per_bucket) << 16) + (i % keys_per_bucket) * 2;
                  });
  return make_keys<Key>(high_bits, low_bits);
}

/**
 * @brief Generates keys that fit in a single roaring container (sparse, even-valued lows)
 */
template <typename Key>
std::vector<Key> make_single_container_keys(uint32_t high_bits = 0, uint32_t num_keys = 200)
{
  auto low_bits =
    std::views::iota(uint32_t{0}, num_keys) | std::views::transform([](auto i) { return i * 2; });
  return make_keys<Key>(high_bits, low_bits);
}

/**
 * @brief Generates a dense run of consecutive keys to trigger run-container encoding
 */
template <typename Key>
std::vector<Key> make_run_container_keys(uint32_t high_bits = 0,
                                         uint32_t run_start = 10,
                                         uint32_t run_size  = 500)
{
  return make_keys<Key>(high_bits, std::views::iota(run_start, run_start + run_size));
}

/**
 * @brief Concatenates `lhs` and `rhs`
 */
template <typename Key>
std::vector<Key> concat_keys(std::vector<Key> lhs, std::vector<Key> const rhs)
{
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
  return lhs;
}

/**
 * @brief Removes the offset table from the first 32-bit bitmap in a serialized payload, producing
 * a portable but non-compliant payload
 */
std::vector<cuda::std::byte> strip_first_no_run_offset_table(cudf::roaring_bitmap_type bitmap_type,
                                                             std::vector<cuda::std::byte> payload)
{
  constexpr uint32_t no_run_cookie    = 12'346;
  constexpr std::size_t key_card_size = sizeof(uint16_t) + sizeof(uint16_t);
  constexpr std::size_t offset_size   = sizeof(uint32_t);
  constexpr std::size_t no_run_prefix = sizeof(uint32_t) + sizeof(uint32_t);

  auto const load_uint32_t = [&](std::size_t offset) {
    uint32_t value;
    std::memcpy(&value, payload.data() + offset, sizeof(value));
    return value;
  };

  auto const bitmap32_offset = bitmap_type == cudf::roaring_bitmap_type::BITS_64
                                 ? sizeof(uint64_t) + sizeof(uint32_t)
                                 : std::size_t{0};
  EXPECT_EQ(load_uint32_t(bitmap32_offset), no_run_cookie);

  // Stripping the offset table to produce a portable but non-compliant payload
  auto const num_containers     = load_uint32_t(bitmap32_offset + sizeof(uint32_t));
  auto const offset_table_begin = bitmap32_offset + no_run_prefix + num_containers * key_card_size;
  auto const offset_table_end   = offset_table_begin + num_containers * offset_size;
  payload.erase(payload.begin() + offset_table_begin, payload.begin() + offset_table_end);
  return payload;
}

/**
 * @brief Verifies that `cudf::roaring_bitmap` correctly queries `expected_keys` given a serialized
 * roaring bitmap `payload`
 */
template <typename Key>
void verify_membership(cudf::host_span<cuda::std::byte const> payload,
                       cudf::host_span<Key const> expected_keys,
                       cudf::host_span<Key const> probe_keys)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto bitmap   = cudf::roaring_bitmap(bitmap_type_v<Key>, payload);
  auto keys_col = cudf::test::fixed_width_column_wrapper<Key>(probe_keys.begin(), probe_keys.end());

  auto result_col = bitmap.contains_async(keys_col, stream, mr);
  auto results    = cudf::detail::make_host_vector_async(
    cudf::device_span<bool const>(result_col->view().template data<bool>(), probe_keys.size()),
    stream);
  stream.synchronize();

  auto const expected_set = std::unordered_set<Key>(expected_keys.begin(), expected_keys.end());
  auto const indices      = std::views::iota(std::size_t{0}, probe_keys.size());
  std::ranges::for_each(indices, [&](std::size_t i) {
    EXPECT_EQ(static_cast<bool>(results[i]), expected_set.contains(probe_keys[i]))
      << "mismatch at key index " << i << " (key=" << probe_keys[i] << ")";
  });
}

/**
 * @brief Checks that the payload is already spec-compliant and queries against it
 */
template <typename Key>
void verify_compliant_payload(cudf::host_span<cuda::std::byte const> payload,
                              cudf::host_span<Key const> expected_keys,
                              cudf::host_span<Key const> probe_keys)
{
  auto const type = bitmap_type_v<Key>;
  auto const payload_sv =
    std::string_view(reinterpret_cast<char const*>(payload.data()), payload.size_bytes());

  EXPECT_TRUE(cudf::iceberg::is_roaring_bitmap_compliant(type, payload_sv));
  auto const compliant = cudf::iceberg::make_compliant_roaring_bitmap(type, payload_sv);
  EXPECT_EQ(compliant, payload_sv);

  verify_membership<Key>(payload, expected_keys, probe_keys);
}

/**
 * @brief Checks that the payload is not spec-compliant, makes it compliant, and queries against it
 */
template <typename Key>
void verify_non_compliant_payload(cudf::host_span<cuda::std::byte const> payload,
                                  cudf::host_span<Key const> expected_keys,
                                  cudf::host_span<Key const> probe_keys)
{
  auto const type = bitmap_type_v<Key>;
  auto const payload_sv =
    std::string_view(reinterpret_cast<char const*>(payload.data()), payload.size_bytes());

  EXPECT_FALSE(cudf::iceberg::is_roaring_bitmap_compliant(type, payload_sv));
  auto const compliant = cudf::iceberg::make_compliant_roaring_bitmap(type, payload_sv);
  EXPECT_NE(compliant, payload_sv);

  EXPECT_TRUE(cudf::iceberg::is_roaring_bitmap_compliant(type, compliant));
  EXPECT_EQ(cudf::iceberg::make_compliant_roaring_bitmap(type, compliant), compliant);

  auto const compliant_span = cudf::host_span<cuda::std::byte const>(
    reinterpret_cast<cuda::std::byte const*>(compliant.data()), compliant.size());
  verify_membership<Key>(compliant_span, expected_keys, probe_keys);
}

}  // namespace

template <typename T>
struct RoaringBitmapComplianceTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(RoaringBitmapComplianceTest, RoaringTypes);

// Many no-run containers, already spec-compliant
TYPED_TEST(RoaringBitmapComplianceTest, AlreadyCompliantManyContainers)
{
  using Key             = TypeParam;
  auto const keys       = make_many_container_keys<Key>();
  auto const probe_keys = concat_keys(keys, make_run_container_keys<Key>());

  auto const serialized = serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/false);

  verify_compliant_payload<Key>(serialized, keys, probe_keys);
}

// Fewer than 4 no-run containers with offset table, already spec-compliant
TYPED_TEST(RoaringBitmapComplianceTest, AlreadyCompliantFewContainers)
{
  using Key = TypeParam;
  // 3 containers of 100 keys each
  auto const keys       = make_many_container_keys<Key>(0, 100, 3);
  auto const probe_keys = concat_keys(keys, make_run_container_keys<Key>());
  auto const serialized = serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/false);

  verify_compliant_payload<Key>(serialized, keys, probe_keys);
}

// No-run + single container + stripped offset table
TYPED_TEST(RoaringBitmapComplianceTest, MissingOffsetTableSingleContainer)
{
  using Key             = TypeParam;
  auto const keys       = make_single_container_keys<Key>();
  auto const probe_keys = concat_keys(keys, make_run_container_keys<Key>());
  auto const serialized = strip_first_no_run_offset_table(
    bitmap_type_v<Key>, serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/false));

  verify_non_compliant_payload<Key>(serialized, keys, probe_keys);
}

// No-run + 5 containers + stripped offset table
TYPED_TEST(RoaringBitmapComplianceTest, MissingOffsetTableManyContainers)
{
  using Key             = TypeParam;
  auto const keys       = make_many_container_keys<Key>(0, 100, 5);
  auto const probe_keys = concat_keys(keys, make_run_container_keys<Key>());
  auto const serialized = strip_first_no_run_offset_table(
    bitmap_type_v<Key>, serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/false));

  verify_non_compliant_payload<Key>(serialized, keys, probe_keys);
}

// Run cookie + single container
TYPED_TEST(RoaringBitmapComplianceTest, RunEncodedSingleContainer)
{
  using Key             = TypeParam;
  auto const keys       = make_run_container_keys<Key>();
  auto const probe_keys = make_single_container_keys<Key>();
  auto const serialized = serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/true);

  verify_compliant_payload<Key>(serialized, keys, probe_keys);
}

// Run cookie + many containers
TYPED_TEST(RoaringBitmapComplianceTest, RunEncodedManyContainers)
{
  using Key                       = TypeParam;
  constexpr auto num_buckets      = uint32_t{4};
  constexpr auto keys_per_bucket  = uint32_t{500};
  constexpr auto probe_per_bucket = uint32_t{600};
  auto const bucket_lo            = [](uint32_t per_bucket) {
    return std::views::iota(uint32_t{0}, num_buckets * per_bucket) |
           std::views::transform(
             [per_bucket](uint32_t k) { return ((k / per_bucket) << 16) + (k % per_bucket); });
  };
  auto const keys       = make_keys<Key>(0, bucket_lo(keys_per_bucket));
  auto const probe_keys = make_keys<Key>(0, bucket_lo(probe_per_bucket));
  auto const serialized = serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/true);

  verify_compliant_payload<Key>(serialized, keys, probe_keys);
}

struct RoaringBitmapComplianceTest64 : public cudf::test::BaseFixture {};

// 64-bit spec-compliant payload with multiple high-32-bit keys. Each embedded 32-bit bitmap is
// spec-compliant
TEST_F(RoaringBitmapComplianceTest64, AlreadyCompliantMultipleHighKeys)
{
  using Key       = cuda::std::uint64_t;
  auto const keys = concat_keys(make_many_container_keys<Key>(0), make_many_container_keys<Key>(1));
  auto const probe_keys = concat_keys(
    keys, concat_keys(make_single_container_keys<Key>(0), make_single_container_keys<Key>(1)));
  auto const serialized = serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/false);

  verify_compliant_payload<Key>(serialized, keys, probe_keys);
}

// 64-bit payload is non-compliant if any embedded 32-bit bitmap is missing offsets
TEST_F(RoaringBitmapComplianceTest64, MissingOffsetTableMultipleHighKeys)
{
  using Key = cuda::std::uint64_t;
  auto const keys =
    concat_keys(make_single_container_keys<Key>(0), make_many_container_keys<Key>(1));
  auto const probe_keys = concat_keys(
    keys, concat_keys(make_run_container_keys<Key>(0), make_run_container_keys<Key>(1)));
  auto const serialized = strip_first_no_run_offset_table(
    bitmap_type_v<Key>, serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/false));

  verify_non_compliant_payload<Key>(serialized, keys, probe_keys);
}

// 64-bit run-encoded payload mixing run-cookie with many-containers
TEST_F(RoaringBitmapComplianceTest64, RunEncodedMultipleHighKeys)
{
  using Key       = cuda::std::uint64_t;
  auto const keys = concat_keys(make_run_container_keys<Key>(0), make_many_container_keys<Key>(1));
  auto const probe_keys =
    concat_keys(make_single_container_keys<Key>(0), make_single_container_keys<Key>(1));
  auto const serialized = serialize_roaring_bitmap<Key>(keys, /*run_optimize=*/true);

  verify_compliant_payload<Key>(serialized, keys, probe_keys);
}
