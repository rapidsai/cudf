/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <join/join_common_utils.cuh>

#include <cudf/detail/join.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <cuco/static_map.cuh>

#include <type_traits>

namespace cudf::detail {

namespace {

using cudf::experimental::row::lhs_index_type;
using cudf::experimental::row::rhs_index_type;

using static_map = cuco::static_map<lhs_index_type,
                                    size_type,
                                    cuda::thread_scope_device,
                                    rmm::mr::stream_allocator_adaptor<default_allocator<char>>>;

/**
 * @brief Check if the given type `T` is a strong index type (i.e., `lhs_index_type` or
 * `rhs_index_type`).
 *
 * @return A boolean value indicating if `T` is a strong index type
 */
template <typename T>
constexpr auto is_strong_index_type()
{
  return std::is_same_v<T, lhs_index_type> || std::is_same_v<T, rhs_index_type>;
}

/**
 * @brief An adapter functor to support strong index types for row hasher that must be operating on
 * `cudf::size_type`.
 */
template <typename Hasher>
struct strong_index_hasher_adapter {
  strong_index_hasher_adapter(Hasher const& hasher) : _hasher{hasher} {}

  template <typename T, CUDF_ENABLE_IF(is_strong_index_type<T>())>
  __device__ constexpr auto operator()(T const idx) const noexcept
  {
    return _hasher(static_cast<size_type>(idx));
  }

 private:
  Hasher const _hasher;
};

/**
 * @brief An adapter functor to support strong index type for table row comparator that must be
 * operating on `cudf::size_type`.
 */
template <typename Comparator>
struct strong_index_comparator_adapter {
  strong_index_comparator_adapter(Comparator const& comparator) : _comparator{comparator} {}

  template <typename T,
            typename U,
            CUDF_ENABLE_IF(is_strong_index_type<T>() && is_strong_index_type<U>())>
  __device__ constexpr auto operator()(T const lhs_index, U const rhs_index) const noexcept
  {
    auto const lhs = static_cast<size_type>(lhs_index);
    auto const rhs = static_cast<size_type>(rhs_index);

    if constexpr (std::is_same_v<T, U> || std::is_same_v<T, lhs_index_type>) {
      return _comparator(lhs, rhs);
    } else {
      // Here we have T == rhs_index_type.
      // This is when the indices are provided in wrong order for two table comparator, so we need
      // to switch them back to the right order before calling the underlying comparator.
      return _comparator(rhs, lhs);
    }
  }

 private:
  Comparator const _comparator;
};

/**
 * @brief Build a row bitmask for the input table.
 *
 * The output bitmask will have invalid bits corresponding to the the input rows having nulls (at
 * any nested level) and vice versa.
 *
 * @param input The input table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of pointer to the output bitmask and the buffer containing the bitmask
 */
std::pair<rmm::device_buffer, bitmask_type const*> build_row_bitmask(table_view const& input,
                                                                     rmm::cuda_stream_view stream)
{
  auto const nullable_columns = get_nullable_columns(input);
  CUDF_EXPECTS(nullable_columns.size() > 0,
               "The input table has nulls thus it should have nullable columns.");

  // If there are more than one nullable column, we compute `bitmask_and` of their null masks.
  // Otherwise, we have only one nullable column and can use its null mask directly.
  if (nullable_columns.size() > 1) {
    auto row_bitmask = cudf::detail::bitmask_and(table_view{nullable_columns}, stream).first;
    auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
    return std::pair(std::move(row_bitmask), row_bitmask_ptr);
  }

  return std::pair(rmm::device_buffer{0, stream}, nullable_columns.front().null_mask());
}

/**
 * @brief Invoke an `operator()` template with a row equality comparator based on the specified
 * `compare_nans` parameter.
 *
 * @param compare_nans The flag to specify whether NaNs should be compared equal or not
 * @param func The input functor to invoke
 */
template <typename Func>
void dispatch_nan_comparator(nan_equality compare_nans, Func&& func)
{
  if (compare_nans == nan_equality::ALL_EQUAL) {
    using nan_equal_comparator =
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
    func(nan_equal_comparator{});
  } else {
    using nan_unequal_comparator = cudf::experimental::row::equality::physical_equality_comparator;
    func(nan_unequal_comparator{});
  }
}

/**
 * @brief Check if rows in the given `needles` table exist in the `haystack` table.
 *
 * This function is designed specifically to work with input tables having lists column(s) at
 * arbitrarily nested levels.
 *
 * @param haystack The table containing the search space
 * @param needles A table of rows whose existence to check in the search space
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param compare_nans Control whether floating-point NaNs values should be compared as equal or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A vector of bools indicating if each row in `needles` has matching rows in `haystack`
 */
rmm::device_uvector<bool> contains_with_lists_or_nans(table_view const& haystack,
                                                      table_view const& needles,
                                                      null_equality compare_nulls,
                                                      nan_equality compare_nans,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  auto map = static_map(compute_hash_table_size(haystack.num_rows()),
                        cuco::empty_key{lhs_index_type{std::numeric_limits<size_type>::max()}},
                        cuco::empty_value{detail::JoinNoneValue},
                        detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value());

  auto const haystack_has_nulls = has_nested_nulls(haystack);
  auto const needles_has_nulls  = has_nested_nulls(needles);
  auto const has_any_nulls      = haystack_has_nulls || needles_has_nulls;

  // Insert row indices of the haystack table as map keys.
  {
    auto const haystack_it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      [] __device__(auto const idx) { return cuco::make_pair(lhs_index_type{idx}, 0); });

    auto const hasher = cudf::experimental::row::hash::row_hasher(haystack, stream);
    auto const d_hasher =
      strong_index_hasher_adapter{hasher.device_hasher(nullate::DYNAMIC{has_any_nulls})};

    auto const comparator = cudf::experimental::row::equality::self_comparator(haystack, stream);

    // If the haystack table has nulls but they are compared unequal, don't insert them.
    // Otherwise, it was known to cause performance issue:
    // - https://github.com/rapidsai/cudf/pull/6943
    // - https://github.com/rapidsai/cudf/pull/8277
    if (haystack_has_nulls && compare_nulls == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(haystack, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

      // Insert only rows that do not have any null at any level.
      auto const insert_map = [&](auto const value_comp) {
        auto const d_eqcomp = strong_index_comparator_adapter{
          comparator.equal_to(nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, value_comp)};
        map.insert_if(haystack_it,
                      haystack_it + haystack.num_rows(),
                      thrust::counting_iterator<size_type>(0),  // stencil
                      row_is_valid{row_bitmask_ptr},
                      d_hasher,
                      d_eqcomp,
                      stream.value());
      };

      dispatch_nan_comparator(compare_nans, insert_map);

    } else {  // haystack_doesn't_have_nulls || compare_nulls == null_equality::EQUAL
      auto const insert_map = [&](auto const value_comp) {
        auto const d_eqcomp = strong_index_comparator_adapter{
          comparator.equal_to(nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, value_comp)};
        map.insert(
          haystack_it, haystack_it + haystack.num_rows(), d_hasher, d_eqcomp, stream.value());
      };

      dispatch_nan_comparator(compare_nans, insert_map);
    }
  }

  // The output vector.
  auto contained = rmm::device_uvector<bool>(needles.num_rows(), stream, mr);

  // Check existence for each row of the needles table in the haystack table.
  {
    auto const needles_it = cudf::detail::make_counting_transform_iterator(
      size_type{0}, [] __device__(auto const idx) { return rhs_index_type{idx}; });

    auto const hasher = cudf::experimental::row::hash::row_hasher(needles, stream);
    auto const d_hasher =
      strong_index_hasher_adapter{hasher.device_hasher(nullate::DYNAMIC{has_any_nulls})};

    auto const comparator =
      cudf::experimental::row::equality::two_table_comparator(haystack, needles, stream);

    auto const check_contains = [&](auto const value_comp) {
      auto const d_eqcomp =
        comparator.equal_to(nullate::DYNAMIC{has_any_nulls}, compare_nulls, value_comp);
      map.contains(needles_it,
                   needles_it + needles.num_rows(),
                   contained.begin(),
                   d_hasher,
                   d_eqcomp,
                   stream.value());
    };

    dispatch_nan_comparator(compare_nans, check_contains);
  }

  return contained;
}

/**
 * @brief Check if rows in the given `needles` table exist in the `haystack` table.
 *
 * This function is designed specifically to work with input tables having only columns of simple
 * types, or structs columns of simple types.
 *
 * @param haystack The table containing the search space
 * @param needles A table of rows whose existence to check in the search space
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A vector of bools indicating if each row in `needles` has matching rows in `haystack`
 */
rmm::device_uvector<bool> contains_without_lists_or_nans(table_view const& haystack,
                                                         table_view const& needles,
                                                         null_equality compare_nulls,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr)
{
  auto map = static_map(compute_hash_table_size(haystack.num_rows()),
                        cuco::empty_key{lhs_index_type{std::numeric_limits<size_type>::max()}},
                        cuco::empty_value{detail::JoinNoneValue},
                        detail::hash_table_allocator_type{default_allocator<char>{}, stream},
                        stream.value());

  auto const haystack_has_nulls = has_nested_nulls(haystack);
  auto const needles_has_nulls  = has_nested_nulls(needles);
  auto const has_any_nulls      = haystack_has_nulls || needles_has_nulls;

  // Flatten the input tables.
  auto const flatten_nullability = has_any_nulls
                                     ? structs::detail::column_nullability::FORCE
                                     : structs::detail::column_nullability::MATCH_INCOMING;
  auto const haystack_flattened_tables =
    structs::detail::flatten_nested_columns(haystack, {}, {}, flatten_nullability);
  auto const needles_flattened_tables =
    structs::detail::flatten_nested_columns(needles, {}, {}, flatten_nullability);
  auto const haystack_flattened = haystack_flattened_tables.flattened_columns();
  auto const needles_flattened  = needles_flattened_tables.flattened_columns();
  auto const haystack_tdv_ptr   = table_device_view::create(haystack_flattened, stream);
  auto const needles_tdv_ptr    = table_device_view::create(needles_flattened, stream);

  // Insert row indices of the haystack table as map keys.
  {
    auto const haystack_it = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      [] __device__(auto const idx) { return cuco::make_pair(lhs_index_type{idx}, 0); });

    auto const d_hasher = strong_index_hasher_adapter{
      row_hash{cudf::nullate::DYNAMIC{has_any_nulls}, *haystack_tdv_ptr}};
    auto const d_eqcomp =
      strong_index_comparator_adapter{row_equality{cudf::nullate::DYNAMIC{haystack_has_nulls},
                                                   *haystack_tdv_ptr,
                                                   *haystack_tdv_ptr,
                                                   compare_nulls}};

    // If the haystack table has nulls but they are compared unequal, don't insert them.
    // Otherwise, it was known to cause performance issue:
    // - https://github.com/rapidsai/cudf/pull/6943
    // - https://github.com/rapidsai/cudf/pull/8277
    if (haystack_has_nulls && compare_nulls == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(haystack, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

      // Insert only rows that do not have any null at any level.
      map.insert_if(haystack_it,
                    haystack_it + haystack.num_rows(),
                    thrust::counting_iterator<size_type>(0),  // stencil
                    row_is_valid{row_bitmask_ptr},
                    d_hasher,
                    d_eqcomp,
                    stream.value());

    } else {  // haystack_doesn't_have_nulls || compare_nulls == null_equality::EQUAL
      map.insert(
        haystack_it, haystack_it + haystack.num_rows(), d_hasher, d_eqcomp, stream.value());
    }
  }

  // The output vector.
  auto contained = rmm::device_uvector<bool>(needles.num_rows(), stream, mr);

  // Check existence for each row of the needles table in the haystack table.
  {
    auto const needles_it = cudf::detail::make_counting_transform_iterator(
      size_type{0}, [] __device__(auto const idx) { return rhs_index_type{idx}; });

    auto const d_hasher = strong_index_hasher_adapter{
      row_hash{cudf::nullate::DYNAMIC{has_any_nulls}, *needles_tdv_ptr}};

    auto const d_eqcomp = strong_index_comparator_adapter{row_equality{
      cudf::nullate::DYNAMIC{has_any_nulls}, *haystack_tdv_ptr, *needles_tdv_ptr, compare_nulls}};

    map.contains(needles_it,
                 needles_it + needles.num_rows(),
                 contained.begin(),
                 d_hasher,
                 d_eqcomp,
                 stream.value());
  }

  return contained;
}

}  // namespace

rmm::device_uvector<bool> contains(table_view const& haystack,
                                   table_view const& needles,
                                   null_equality compare_nulls,
                                   nan_equality compare_nans,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  // Checking for only one table is enough, because both tables will be checked to have the same
  // shape later during row comparisons.
  auto const has_lists = std::any_of(haystack.begin(), haystack.end(), [](auto const& col) {
    return cudf::structs::detail::is_or_has_nested_lists(col);
  });

  if (has_lists || compare_nans == nan_equality::UNEQUAL) {
    // We must call a separate code path that uses the new experimental row hasher and row
    // comparator if:
    //  - The input has lists column, or
    //  - Floating-point NaNs are compared as unequal.
    // Inputs with these conditions are supported only by this code path.
    return contains_with_lists_or_nans(haystack, needles, compare_nulls, compare_nans, stream, mr);
  }

  // If the input tables don't have lists column and NaNs are compared equal, we rely on the classic
  // code path that flattens the input tables for row comparisons. This way is known to have
  // better performance.
  return contains_without_lists_or_nans(haystack, needles, compare_nulls, stream, mr);

  // Note: We have to keep separate code paths because unifying them will cause performance
  // regression for the input having no nested lists.
  //
  // TODO: We should unify these code paths in the future when performance regression is no longer
  // happening.
}

}  // namespace cudf::detail
