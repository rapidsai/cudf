/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <roaring/roaring.h>
#include <roaring/roaring64.h>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <vector>

namespace cudf::test {

/**
 * @brief Serializes a 32-bit roaring bitmap containing the given keys
 */
inline auto serialize_roaring32(std::vector<cuda::std::uint32_t> const& keys)
{
  auto* bitmap = roaring::api::roaring_bitmap_create();
  auto ctx     = roaring::api::roaring_bulk_context_t{};
  for (auto key : keys) {
    roaring::api::roaring_bitmap_add_bulk(bitmap, &ctx, key);
  }

  auto const num_bytes = roaring::api::roaring_bitmap_portable_size_in_bytes(bitmap);
  auto serialized      = std::vector<cuda::std::byte>(num_bytes);
  roaring::api::roaring_bitmap_portable_serialize(
    bitmap, reinterpret_cast<char*>(serialized.data()));
  roaring::api::roaring_bitmap_free(bitmap);
  return serialized;
}

/**
 * @brief Serializes a 64-bit roaring bitmap containing the given keys
 */
inline auto serialize_roaring64(std::vector<cuda::std::uint64_t> const& keys)
{
  auto* bitmap = roaring::api::roaring64_bitmap_create();
  auto ctx     = roaring::api::roaring64_bulk_context_t{.high_bytes = {0, 0, 0, 0, 0, 0},
                                                        .leaf       = nullptr};
  for (auto key : keys) {
    roaring::api::roaring64_bitmap_add_bulk(bitmap, &ctx, key);
  }

  auto const num_bytes = roaring::api::roaring64_bitmap_portable_size_in_bytes(bitmap);
  auto serialized      = std::vector<cuda::std::byte>(num_bytes);
  roaring::api::roaring64_bitmap_portable_serialize(
    bitmap, reinterpret_cast<char*>(serialized.data()));
  roaring::api::roaring64_bitmap_free(bitmap);
  return serialized;
}

}  // namespace cudf::test
