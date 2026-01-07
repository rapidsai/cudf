/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/hyperloglog.cuh>
#include <cuda/functional>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief HyperLogLog-based approximate distinct count sketch for use by the public API.
 *
 * This detail implementation provides the core HyperLogLog functionality used by the
 * public `cudf::approx_distinct_count` class. It uses XXHash64 for hashing table rows
 * and maintains a cuco::hyperloglog sketch for cardinality estimation.
 */
class approx_distinct_count {
 public:
  /**
   * @brief Construct an approximate distinct count sketch from a table.
   *
   * @param input Table whose rows will be added to the sketch
   * @param precision The precision parameter for HyperLogLog (4-18). Higher precision gives
   *                  better accuracy but uses more memory.
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL`
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  approx_distinct_count(table_view const& input,
                        std::int32_t precision,
                        null_policy null_handling,
                        nan_policy nan_handling,
                        rmm::cuda_stream_view stream);

  /**
   * @brief Construct an approximate distinct count sketch from serialized sketch bytes.
   *
   * @param sketch_span The serialized sketch bytes to reconstruct from
   * @param precision The precision parameter that was used to create the sketch (4-18)
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  approx_distinct_count(cuda::std::span<cuda::std::byte> sketch_span,
                        std::int32_t precision,
                        rmm::cuda_stream_view stream);

  approx_distinct_count() = delete;
  ~approx_distinct_count();
  approx_distinct_count(approx_distinct_count const&)            = delete;
  approx_distinct_count& operator=(approx_distinct_count const&) = delete;
  approx_distinct_count(approx_distinct_count&&) noexcept        = default;
  approx_distinct_count& operator=(approx_distinct_count&&)      = delete;

  /**
   * @brief Add rows from a table to the sketch.
   *
   * @param input Table whose rows will be added
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL`
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void add(table_view const& input,
           null_policy null_handling,
           nan_policy nan_handling,
           rmm::cuda_stream_view stream);

  /**
   * @brief Merge another sketch into this sketch.
   *
   * @param other The sketch to merge into this sketch
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void merge(approx_distinct_count const& other, rmm::cuda_stream_view stream);

  /**
   * @brief Merge a sketch from raw bytes into this sketch.
   *
   * @warning It is the caller's responsibility to ensure that the provided sketch span was created
   * with the same approx_distinct_count configuration (precision, null/NaN handling, etc.) as this
   * sketch. Merging incompatible sketches will produce incorrect results.
   *
   * @param sketch_span The sketch bytes to merge into this sketch
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void merge(cuda::std::span<cuda::std::byte> sketch_span, rmm::cuda_stream_view stream);

  /**
   * @brief Get the raw sketch bytes.
   *
   * @return A span view of the sketch bytes
   */
  [[nodiscard]] cuda::std::span<cuda::std::byte> sketch() noexcept;

  /**
   * @brief Estimate the approximate number of distinct rows in the sketch.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Approximate number of distinct rows
   */
  [[nodiscard]] cudf::size_type estimate(rmm::cuda_stream_view stream) const;

 private:
  using hll_type = cuco::hyperloglog<uint64_t,
                                     cuda::thread_scope_device,
                                     cuda::std::identity,
                                     rmm::mr::polymorphic_allocator<cuda::std::byte>>;
  hll_type _impl;
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
