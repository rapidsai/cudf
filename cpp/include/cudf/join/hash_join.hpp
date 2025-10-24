/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>
#include <utility>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

// forward declaration
namespace hashing::detail {
/**
 * @brief Forward declaration for our Murmur Hash 3 implementation
 */
template <typename T>
class MurmurHash3_x86_32;
}  // namespace hashing::detail

namespace detail {
/**
 * @brief Forward declaration for our hash join
 */
template <typename T>
class hash_join;
}  // namespace detail

/**
 * @brief The enum class to specify if any of the input join tables (`build` table and any later
 * `probe` table) has nulls.
 *
 * This is used upon hash_join object construction to specify the existence of nulls in all the
 * possible input tables. If such null existence is unknown, `YES` should be used as the default
 * option.
 */
enum class nullable_join : bool { YES, NO };

/**
 * @brief Hash join that builds hash table in creation and probes results in subsequent `*_join`
 * member functions.
 *
 * This class enables the hash join scheme that builds hash table once, and probes as many times as
 * needed (possibly in parallel).
 */
class hash_join {
 public:
  using impl_type = typename cudf::detail::hash_join<
    cudf::hashing::detail::MurmurHash3_x86_32<cudf::hash_value_type>>;  ///< Implementation type

  hash_join() = delete;
  ~hash_join();
  hash_join(hash_join const&)            = delete;
  hash_join(hash_join&&)                 = delete;
  hash_join& operator=(hash_join const&) = delete;
  hash_join& operator=(hash_join&&)      = delete;

  /**
   * @brief Construct a hash join object for subsequent probe calls.
   *
   * @note The `hash_join` object must not outlive the table viewed by `build`, else behavior is
   * undefined.
   *
   * @throws std::invalid_argument if the build table has no columns
   *
   * @param build The build table, from which the hash table is built
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  hash_join(cudf::table_view const& build,
            null_equality compare_nulls,
            rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @copydoc hash_join(cudf::table_view const&, null_equality, rmm::cuda_stream_view)
   *
   * @throws std::invalid_argument if load_factor is not greater than 0 and less than or equal to 1
   *
   * @param has_nulls Flag to indicate if there exists any nulls in the `build` table or
   *                  any `probe` table that will be used later for join
   * @param load_factor The hash table occupancy ratio in (0,1]. A value of 0.5 means 50% desired
   * occupancy.
   */
  hash_join(cudf::table_view const& build,
            nullable_join has_nulls,
            null_equality compare_nulls,
            double load_factor,
            rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing an inner join between two tables with `build` and `probe`
   * as the join keys .
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& probe,
             std::optional<std::size_t> output_size = {},
             rmm::cuda_stream_view stream           = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the row indices that can be used to construct the result of performing
   * a left join between two tables. @see cudf::left_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing a left join between two tables with `build` and `probe`
   * as the join keys.
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& probe,
            std::optional<std::size_t> output_size = {},
            rmm::cuda_stream_view stream           = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the row indices that can be used to construct the result of performing
   * a full join between two tables. @see cudf::full_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing a full join between two tables with `build` and `probe`
   * as the join keys .
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& probe,
            std::optional<std::size_t> output_size = {},
            rmm::cuda_stream_view stream           = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the exact number of matches (rows) when performing an inner join with the specified
   * probe table.
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The exact number of output when performing an inner join between two tables with
   * `build` and `probe` as the join keys .
   */
  [[nodiscard]] std::size_t inner_join_size(
    cudf::table_view const& probe, rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

  /**
   * Returns the exact number of matches (rows) when performing a left join with the specified probe
   * table.
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The exact number of output when performing a left join between two tables with `build`
   * and `probe` as the join keys .
   */
  [[nodiscard]] std::size_t left_join_size(
    cudf::table_view const& probe, rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

  /**
   * Returns the exact number of matches (rows) when performing a full join with the specified probe
   * table.
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the intermediate table and columns' device
   * memory.
   *
   * @return The exact number of output when performing a full join between two tables with `build`
   * and `probe` as the join keys .
   */
  [[nodiscard]] std::size_t full_join_size(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns context information about matches between the probe and build tables.
   *
   * This method computes, for each row in the probe table, how many matching rows exist in
   * the build table according to inner join semantics, and returns the number of matches through a
   * join_match_context object.
   *
   * This is particularly useful for:
   * - Determining the total size of a potential join result without materializing it
   * - Planning partitioned join operations for large datasets
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table to join with the pre-processed build table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A join_match_context object containing the probe table view and a device vector
   *         of match counts for each row in the probe table
   */
  [[nodiscard]] cudf::join_match_context inner_join_match_context(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns context information about matches between the probe and build tables.
   *
   * This method computes, for each row in the probe table, how many matching rows exist in
   * the build table according to left join semantics, and returns the number of matches through a
   * join_match_context object.
   *
   * For left join, every row in the probe table will have at least one match (either with a
   * matching row from the build table or with a null placeholder).
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table to join with the pre-processed build table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A join_match_context object containing the probe table view and a device vector
   *         of match counts for each row in the probe table
   */
  [[nodiscard]] cudf::join_match_context left_join_match_context(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns context information about matches between the probe and build tables.
   *
   * This method computes, for each row in the probe table, how many matching rows exist in
   * the build table according to full join semantics, and returns the number of matches through a
   * join_match_context object.
   *
   * For full join, this includes matches for probe table rows, and the result may need to be
   * combined with unmatched rows from the build table to get the complete picture.
   *
   * @throw std::invalid_argument If the input probe table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param probe The probe table to join with the pre-processed build table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A join_match_context object containing the probe table view and a device vector
   *         of match counts for each row in the probe table
   */
  [[nodiscard]] cudf::join_match_context full_join_match_context(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<impl_type const> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
