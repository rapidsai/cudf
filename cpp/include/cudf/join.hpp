/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {

/**
 * @brief Enum to indicate whether the distinct join table has nested columns or not
 *
 * @ingroup column_join
 */
enum class has_nested : bool { YES, NO };

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

/**
 * @brief Forward declaration for our distinct hash join
 */
template <cudf::has_nested HasNested>
class distinct_hash_join;
}  // namespace detail

/**
 * @addtogroup column_join
 * @{
 * @file
 */

/**
 * @brief Returns a pair of row index vectors corresponding to an
 * inner join between the specified tables.
 *
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{1, 2}, {0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing an inner join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(cudf::table_view const& left_keys,
           cudf::table_view const& right_keys,
           null_equality compare_nulls       = null_equality::EQUAL,
           rmm::cuda_stream_view stream      = cudf::get_default_stream(),
           rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to a
 * left join between the specified tables.
 *
 * The first returned vector contains all the row indices from the left
 * table (in unspecified order). The corresponding value in the
 * second returned vector is either (1) the row index of the matched row
 * from the right table, if there is a match  or  (2) an unspecified
 * out-of-bounds value.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{0, 1, 2}, {None, 0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{0, 1, 2}, {None, 0, None}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a left join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
left_join(cudf::table_view const& left_keys,
          cudf::table_view const& right_keys,
          null_equality compare_nulls       = null_equality::EQUAL,
          rmm::cuda_stream_view stream      = cudf::get_default_stream(),
          rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to a
 * full join between the specified tables.
 *
 * Taken pairwise, the values from the returned vectors are one of:
 * (1) row indices corresponding to matching rows from the left and
 * right tables, (2) a row index and an unspecified out-of-bounds value,
 * representing a row from one table without a match in the other.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Result: {{0, 1, 2, None}, {None, 0, 1, 2}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Result: {{0, 1, 2, None, None}, {None, 0, None, 1, 2}}
 * @endcode
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a full join between two tables with `left_keys` and `right_keys`
 * as the join keys .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
full_join(cudf::table_view const& left_keys,
          cudf::table_view const& right_keys,
          null_equality compare_nulls       = null_equality::EQUAL,
          rmm::cuda_stream_view stream      = cudf::get_default_stream(),
          rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a vector of row indices corresponding to a left semi-join
 * between the specified tables.
 *
 * The returned vector contains the row indices from the left table
 * for which there is a matching row in the right table.
 *
 * @code{.pseudo}
 * TableA: {{0, 1, 2}}
 * TableB: {{1, 2, 3}}
 * Result: {1, 2}
 * @endcode
 *
 * @param left_keys The left table
 * @param right_keys The right table
 * @param compare_nulls Controls whether null join-key values should match or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector `left_indices` that can be used to construct
 * the result of performing a left semi join between two tables with
 * `left_keys` and `right_keys` as the join keys .
 */
std::unique_ptr<rmm::device_uvector<size_type>> left_semi_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a vector of row indices corresponding to a left anti join
 * between the specified tables.
 *
 * The returned vector contains the row indices from the left table
 * for which there is no matching row in the right table.
 *
 * @code{.pseudo}
 * TableA: {{0, 1, 2}}
 * TableB: {{1, 2, 3}}
 * Result: {0}
 * @endcode
 *
 * @throw cudf::logic_error if the number of columns in either `left_keys` or `right_keys` is 0
 *
 * @param[in] left_keys The left table
 * @param[in] right_keys The right table
 * @param[in] compare_nulls controls whether null join-key values
 * should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A column `left_indices` that can be used to construct
 * the result of performing a left anti join between two tables with
 * `left_keys` and `right_keys` as the join keys .
 */
std::unique_ptr<rmm::device_uvector<size_type>> left_anti_join(
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Performs a cross join on two tables (`left`, `right`)
 *
 * The cross join returns the cartesian product of rows from each table.
 *
 * @note Warning: This function can easily cause out-of-memory errors. The size of the output is
 * equal to `left.num_rows() * right.num_rows()`. Use with caution.
 *
 * @code{.pseudo}
 * Left a: {0, 1, 2}
 * Right b: {3, 4, 5}
 * Result: { a: {0, 0, 0, 1, 1, 1, 2, 2, 2}, b: {3, 4, 5, 3, 4, 5, 3, 4, 5} }
 * @endcode

 * @throw cudf::logic_error if the number of columns in either `left` or `right` table is 0
 *
 * @param left  The left table
 * @param right The right table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr    Device memory resource used to allocate the returned table's device memory
 *
 * @return     Result of cross joining `left` and `right` tables
 */
std::unique_ptr<cudf::table> cross_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

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
   * @param has_nulls Flag to indicate if there exists any nulls in the `build` table or
   *        any `probe` table that will be used later for join
   */
  hash_join(cudf::table_view const& build,
            nullable_join has_nulls,
            null_equality compare_nulls,
            rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join(). Behavior is undefined if the
   * provided `output_size` is smaller than the actual output size.
   *
   * @param probe The probe table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @throw cudf::logic_error If the input probe table has nulls while this hash_join object was not
   * constructed with null check.
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
   * @param probe The probe table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @throw cudf::logic_error If the input probe table has nulls while this hash_join object was not
   * constructed with null check.
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
   * @param probe The probe table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @throw cudf::logic_error If the input probe table has nulls while this hash_join object was not
   * constructed with null check.
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
   * @param probe The probe table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @throw cudf::logic_error If the input probe table has nulls while this hash_join object was not
   * constructed with null check.
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
   * @param probe The probe table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @throw cudf::logic_error If the input probe table has nulls while this hash_join object was not
   * constructed with null check.
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
   * @param probe The probe table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the intermediate table and columns' device
   * memory.
   *
   * @throw cudf::logic_error If the input probe table has nulls while this hash_join object was not
   * constructed with null check.
   *
   * @return The exact number of output when performing a full join between two tables with `build`
   * and `probe` as the join keys .
   */
  [[nodiscard]] std::size_t full_join_size(
    cudf::table_view const& probe,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  const std::unique_ptr<impl_type const> _impl;
};

/**
 * @brief Distinct hash join that builds hash table in creation and probes results in subsequent
 * `*_join` member functions
 *
 * @note Behavior is undefined if the build table contains duplicates.
 * @note All NaNs are considered as equal
 *
 * @tparam HasNested Flag indicating whether there are nested columns in build/probe table
 */
// TODO: `HasNested` to be removed via dispatching
template <cudf::has_nested HasNested>
class distinct_hash_join {
 public:
  distinct_hash_join() = delete;
  ~distinct_hash_join();
  distinct_hash_join(distinct_hash_join const&)            = delete;
  distinct_hash_join(distinct_hash_join&&)                 = delete;
  distinct_hash_join& operator=(distinct_hash_join const&) = delete;
  distinct_hash_join& operator=(distinct_hash_join&&)      = delete;

  /**
   * @brief Constructs a distinct hash join object for subsequent probe calls
   *
   * @param build The build table that contains distinct elements
   * @param probe The probe table, from which the keys are probed
   * @param has_nulls Flag to indicate if there exists any nulls in the `build` table or
   *        any `probe` table that will be used later for join
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  distinct_hash_join(cudf::table_view const& build,
                     cudf::table_view const& probe,
                     nullable_join has_nulls      = nullable_join::YES,
                     null_equality compare_nulls  = null_equality::EQUAL,
                     rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join().
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices' device memory.
   *
   * @return A pair of columns [`build_indices`, `probe_indices`] that can be used to
   * construct the result of performing an inner join between two tables
   * with `build` and `probe` as the join keys.
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(rmm::cuda_stream_view stream      = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the build table indices that can be used to construct the result of performing
   * a left join between two tables.
   *
   * @note For a given row index `i` of the probe table, the resulting `build_indices[i]` contains
   * the row index of the matched row from the build table if there is a match. Otherwise, contains
   * `JoinNoneValue`.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   * @return A `build_indices` column that can be used to construct the result of
   * performing a left join between two tables with `build` and `probe` as the join
   * keys.
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> left_join(
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  using impl_type = typename cudf::detail::distinct_hash_join<HasNested>;  ///< Implementation type

  std::unique_ptr<impl_type> _impl;  ///< Distinct hash join implementation
};

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs
 * of rows between the specified tables where the predicate evaluates to true.
 *
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {{1, 2}, {0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a conditional inner join between two tables `left` and `right` .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_inner_join(table_view const& left,
                       table_view const& right,
                       ast::expression const& binary_predicate,
                       std::optional<std::size_t> output_size = {},
                       rmm::cuda_stream_view stream           = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs
 * of rows between the specified tables where the predicate evaluates to true,
 * or null matches for rows in left that have no match in right.
 *
 * The first returned vector contains all the row indices from the left
 * table (in unspecified order). The corresponding value in the
 * second returned vector is either (1) the row index of the matched row
 * from the right table, if there is a match or (2) an unspecified
 * out-of-bounds value.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {{0, 1, 2}, {None, 0, 1}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {{0, 1, 2}, {None, 0, None}}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a conditional left join between two tables `left` and `right` .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_left_join(table_view const& left,
                      table_view const& right,
                      ast::expression const& binary_predicate,
                      std::optional<std::size_t> output_size = {},
                      rmm::cuda_stream_view stream           = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs
 * of rows between the specified tables where the predicate evaluates to true,
 * or null matches for rows in either table that have no match in the other.
 *
 * Taken pairwise, the values from the returned vectors are one of:
 * (1) row indices corresponding to matching rows from the left and
 * right tables, (2) a row index and an unspecified out-of-bounds value,
 * representing a row from one table without a match in the other.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {{0, 1, 2, None}, {None, 0, 1, 2}}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {{0, 1, 2, None, None}, {None, 0, None, 1, 2}}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a conditional full join between two tables `left` and `right` .
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_full_join(table_view const& left,
                      table_view const& right,
                      ast::expression const& binary_predicate,
                      rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to all rows in the left table
 * for which there exists some row in the right table where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {1, 2}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {1}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector `left_indices` that can be used to construct the result of
 * performing a conditional left semi join between two tables `left` and
 * `right` .
 */
std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_semi_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size = {},
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to all rows in the left table
 * for which there does not exist any row in the right table where the
 * predicate evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @code{.pseudo}
 * Left: {{0, 1, 2}}
 * Right: {{1, 2, 3}}
 * Expression: Left.Column_0 == Right.Column_0
 * Result: {0}
 *
 * Left: {{0, 1, 2}, {3, 4, 5}}
 * Right: {{1, 2, 3}, {4, 6, 7}}
 * Expression: (Left.Column_0 == Right.Column_0) AND (Left.Column_1 == Right.Column_1)
 * Result: {0, 2}
 * @endcode
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A vector `left_indices` that can be used to construct the result of
 * performing a conditional left anti join between two tables `left` and
 * `right` .
 */
std::unique_ptr<rmm::device_uvector<size_type>> conditional_left_anti_join(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  std::optional<std::size_t> output_size = {},
  rmm::cuda_stream_view stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs of
 * rows between the specified tables where the columns of the equality table
 * are equal and the predicate evaluates to true on the conditional tables.
 *
 * The first returned vector contains the row indices from the left
 * table that have a match in the right table (in unspecified order).
 * The corresponding values in the second returned vector are
 * the matched row indices from the right table.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output. It is the user's
 * responsibility to choose a suitable compare_nulls value AND use appropriate
 * null-safe operators in the expression.
 *
 * If the provided output size or per-row counts are incorrect, behavior is undefined.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {{1}, {0}}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param output_size_data An optional pair of values indicating the exact output size and the
 * number of matches for each row in the larger of the two input tables, left or right (may be
 * precomputed using the corresponding mixed_inner_join_size API).
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a mixed inner join between the four input tables.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls = null_equality::EQUAL,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> output_size_data = {},
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs of
 * rows between the specified tables where the columns of the equality table
 * are equal and the predicate evaluates to true on the conditional tables,
 * or null matches for rows in left that have no match in right.
 *
 * The first returned vector contains the row indices from the left
 * tables that have a match in the right tables (in unspecified order).
 * The corresponding value in the second returned vector is either (1)
 * the row index of the matched row from the right tables, or (2) an
 * unspecified out-of-bounds value.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output. It is the user's
 * responsibility to choose a suitable compare_nulls value AND use appropriate
 * null-safe operators in the expression.
 *
 * If the provided output size or per-row counts are incorrect, behavior is undefined.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {{0, 1, 2}, {None, 0, None}}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param output_size_data An optional pair of values indicating the exact output size and the
 * number of matches for each row in the larger of the two input tables, left or right (may be
 * precomputed using the corresponding mixed_left_join_size API).
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a mixed left join between the four input tables.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_left_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls = null_equality::EQUAL,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> output_size_data = {},
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a pair of row index vectors corresponding to all pairs of
 * rows between the specified tables where the columns of the equality table
 * are equal and the predicate evaluates to true on the conditional tables,
 * or null matches for rows in either pair of tables that have no matches in
 * the other pair.
 *
 * Taken pairwise, the values from the returned vectors are one of:
 * (1) row indices corresponding to matching rows from the left and
 * right tables, (2) a row index and an unspecified out-of-bounds value,
 * representing a row from one table without a match in the other.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output. It is the user's
 * responsibility to choose a suitable compare_nulls value AND use appropriate
 * null-safe operators in the expression.
 *
 * If the provided output size or per-row counts are incorrect, behavior is undefined.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {{0, 1, 2, None, None}, {None, 0, None, 1, 2}}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param output_size_data An optional pair of values indicating the exact output size and the
 * number of matches for each row in the larger of the two input tables, left or right (may be
 * precomputed using the corresponding mixed_full_join_size API).
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a mixed full join between the four input tables.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_full_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls = null_equality::EQUAL,
  std::optional<std::pair<std::size_t, device_span<size_type const>>> output_size_data = {},
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to all rows in the left tables
 * where the columns of the equality table are equal and the predicate
 * evaluates to true on the conditional tables.
 *
 * If the provided predicate returns NULL for a pair of rows (left, right), the
 * left row is not included in the output. It is the user's responsibility to
 * choose a suitable compare_nulls value AND use appropriate null-safe
 * operators in the expression.
 *
 * If the provided output size or per-row counts are incorrect, behavior is undefined.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {1}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a mixed full join between the four input tables.
 */
std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_semi_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns an index vector corresponding to all rows in the left tables
 * for which there is no row in the right tables where the columns of the
 * equality table are equal and the predicate evaluates to true on the
 * conditional tables.
 *
 * If the provided predicate returns NULL for a pair of rows (left, right), the
 * left row is not included in the output. It is the user's responsibility to
 * choose a suitable compare_nulls value AND use appropriate null-safe
 * operators in the expression.
 *
 * If the provided output size or per-row counts are incorrect, behavior is undefined.
 *
 * @code{.pseudo}
 * left_equality: {{0, 1, 2}}
 * right_equality: {{1, 2, 3}}
 * left_conditional: {{4, 4, 4}}
 * right_conditional: {{3, 4, 5}}
 * Expression: Left.Column_0 > Right.Column_0
 * Result: {0, 2}
 * @endcode
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct
 * the result of performing a mixed full join between the four input tables.
 */
std::unique_ptr<rmm::device_uvector<size_type>> mixed_left_anti_join(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * mixed inner join between the specified tables where the columns of the
 * equality table are equal and the predicate evaluates to true on the
 * conditional tables.
 *
 * If the provided predicate returns NULL for a pair of rows (left, right),
 * that pair is not included in the output. It is the user's responsibility to
 * choose a suitable compare_nulls value AND use appropriate null-safe
 * operators in the expression.
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair containing the size that would result from performing the
 * requested join and the number of matches for each row in one of the two
 * tables. Which of the two tables is an implementation detail and should not
 * be relied upon, simply passed to the corresponding `mixed_inner_join` API as
 * is.
 */
std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_inner_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * mixed left join between the specified tables where the columns of the
 * equality table are equal and the predicate evaluates to true on the
 * conditional tables.
 *
 * If the provided predicate returns NULL for a pair of rows (left, right),
 * that pair is not included in the output. It is the user's responsibility to
 * choose a suitable compare_nulls value AND use appropriate null-safe
 * operators in the expression.
 *
 * @throw cudf::data_type_error If the binary predicate outputs a non-boolean result.
 * @throw cudf::logic_error If the number of rows in left_equality and left_conditional do not
 * match.
 * @throw cudf::logic_error If the number of rows in right_equality and right_conditional do not
 * match.
 *
 * @param left_equality The left table used for the equality join
 * @param right_equality The right table used for the equality join
 * @param left_conditional The left table used for the conditional join
 * @param right_conditional The right table used for the conditional join
 * @param binary_predicate The condition on which to join
 * @param compare_nulls Whether or not null values join to each other or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return A pair containing the size that would result from performing the
 * requested join and the number of matches for each row in one of the two
 * tables. Which of the two tables is an implementation detail and should not
 * be relied upon, simply passed to the corresponding `mixed_left_join` API as
 * is.
 */
std::pair<std::size_t, std::unique_ptr<rmm::device_uvector<size_type>>> mixed_left_join_size(
  table_view const& left_equality,
  table_view const& right_equality,
  table_view const& left_conditional,
  table_view const& right_conditional,
  ast::expression const& binary_predicate,
  null_equality compare_nulls       = null_equality::EQUAL,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * conditional inner join between the specified tables where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return The size that would result from performing the requested join
 */
std::size_t conditional_inner_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * conditional left join between the specified tables where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return The size that would result from performing the requested join
 */
std::size_t conditional_left_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * conditional left semi join between the specified tables where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return The size that would result from performing the requested join
 */
std::size_t conditional_left_semi_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the exact number of matches (rows) when performing a
 * conditional left anti join between the specified tables where the predicate
 * evaluates to true.
 *
 * If the provided predicate returns NULL for a pair of rows
 * (left, right), that pair is not included in the output.
 *
 * @throw cudf::data_type_error if the binary predicate outputs a non-boolean result.
 *
 * @param left The left table
 * @param right The right table
 * @param binary_predicate The condition on which to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table and columns' device memory
 *
 * @return The size that would result from performing the requested join
 */
std::size_t conditional_left_anti_join_size(
  table_view const& left,
  table_view const& right,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
