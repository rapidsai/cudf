/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/span>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace CUDF_EXPORT cudf {

// Forward declaration
namespace detail {
struct approx_distinct_count;
}

/**
 * @brief Object-oriented HyperLogLog sketch for approximate distinct counting.
 *
 * This class provides an object-oriented interface to HyperLogLog sketches, allowing
 * incremental addition of data and cardinality estimation.
 *
 * The implementation uses XXHash64 to hash table rows into 64-bit values, which are
 * then added to the HyperLogLog sketch without additional hashing (identity function).
 *
 * @par HyperLogLog Precision Parameter
 * The precision parameter (p) is the number of bits used to index into the register array.
 * It determines the number of registers (m = 2^p) in the HLL sketch:
 * - Memory usage: 2^p * 4 bytes (m registers of 4 bytes each for GPU atomics)
 * - Standard error: 1.04 / sqrt(m) = 1.04 / sqrt(2^p)
 *
 * Common precision values:
 * - p = 10: m = 1,024 registers, ~3.2% standard error, 4KB memory
 * - p = 12 (default): m = 4,096 registers, ~1.6% standard error, 16KB memory
 * - p = 14: m = 16,384 registers, ~0.8% standard error, 64KB memory
 * - p = 16: m = 65,536 registers, ~0.4% standard error, 256KB memory
 *
 * Valid range: p ∈ [4, 18]. This is not a hard theoretical limit but an empirically
 * recommended range:
 * - Below 4: Too few registers for HLL's statistical assumptions, resulting in high
 *   variance and unstable estimates.
 * - Above 18: Rapidly diminishing accuracy gains while incurring significant memory
 *   growth, making the structure no longer space-efficient for approximate counting.
 *
 * This range represents a practical engineering compromise from HLL++ and is widely
 * adopted by systems such as Apache Spark. The default of 12 aligns with Spark's
 * configuration and is the largest precision that fits efficiently in GPU shared memory,
 * enabling optimal performance for our implementation.
 *
 * Example usage:
 * @code{.cpp}
 *   auto adc = cudf::approx_distinct_count(table1);
 *   auto count1 = adc.estimate();
 *
 *   adc.add(table2);
 *   auto count2 = adc.estimate();
 * @endcode
 */
class approx_distinct_count {
 public:
  using impl_type = cudf::detail::approx_distinct_count;  ///< Implementation type

  /**
   * @brief Construct an approximate distinct count sketch from a table.
   *
   * @param input Table whose rows will be added to the sketch
   * @param precision The precision parameter for HyperLogLog (4-18). Higher precision gives
   *                  better accuracy but uses more memory. Default is 12.
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls (default: `EXCLUDE`)
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL` (default: `NAN_IS_NULL`)
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  approx_distinct_count(table_view const& input,
                        std::int32_t precision       = 12,
                        null_policy null_handling    = null_policy::EXCLUDE,
                        nan_policy nan_handling      = nan_policy::NAN_IS_NULL,
                        rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Construct an approximate distinct count sketch from serialized sketch bytes.
   *
   * This constructor enables distributed distinct counting by allowing sketches to be
   * constructed from serialized data. The sketch data is copied into the newly created
   * object, which then owns its own independent storage.
   *
   * @warning The precision parameter must match the precision used to create the original
   * sketch. The size of the sketch span must be exactly 2^precision bytes. Providing
   * an incompatible sketch will produce incorrect results or errors.
   *
   * @param sketch_span The serialized sketch bytes to reconstruct from
   * @param precision The precision parameter that was used to create the sketch (4-18)
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  approx_distinct_count(cuda::std::span<cuda::std::byte> sketch_span,
                        std::int32_t precision,
                        rmm::cuda_stream_view stream = cudf::get_default_stream());

  ~approx_distinct_count();

  approx_distinct_count(approx_distinct_count const&)            = delete;
  approx_distinct_count& operator=(approx_distinct_count const&) = delete;
  /** @brief Default move constructor */
  approx_distinct_count(approx_distinct_count&&) = default;
  /**
   * @brief Default move assignment operator
   * @return Reference to this object
   */
  approx_distinct_count& operator=(approx_distinct_count&&) = default;

  /**
   * @brief Add rows from a table to the sketch.
   *
   * @param input Table whose rows will be added
   * @param null_handling `INCLUDE` or `EXCLUDE` rows with nulls (default: `EXCLUDE`)
   * @param nan_handling `NAN_IS_VALID` or `NAN_IS_NULL` (default: `NAN_IS_NULL`)
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void add(table_view const& input,
           null_policy null_handling    = null_policy::EXCLUDE,
           nan_policy nan_handling      = nan_policy::NAN_IS_NULL,
           rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Merge another sketch into this sketch.
   *
   * After merging, this sketch will contain the combined distinct count estimate of both sketches.
   *
   * @param other The sketch to merge into this sketch
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void merge(approx_distinct_count const& other,
             rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Merge a sketch from raw bytes into this sketch.
   *
   * This allows merging sketches that have been serialized or created elsewhere, enabling
   * distributed distinct counting scenarios.
   *
   * @warning It is the caller's responsibility to ensure that the provided sketch span was created
   * with the same approx_distinct_count configuration (precision, null/NaN handling, etc.) as this
   * sketch. Merging incompatible sketches will produce incorrect results.
   *
   * @param sketch_span The sketch bytes to merge into this sketch
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void merge(cuda::std::span<cuda::std::byte> sketch_span,
             rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Get the raw sketch bytes for serialization or external merging.
   *
   * The returned span provides access to the internal sketch storage.
   * This can be used to serialize the sketch, transfer it between processes,
   * or merge it with other sketches using the span-based merge API.
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
  [[nodiscard]] std::size_t estimate(
    rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

 private:
  std::unique_ptr<impl_type> _impl;
};

}  // namespace CUDF_EXPORT cudf
