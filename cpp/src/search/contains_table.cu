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

#include <cudf/detail/null_mask.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <cuco/static_set.cuh>

#include <type_traits>

namespace cudf::detail {

namespace {

using cudf::experimental::row::lhs_index_type;
using cudf::experimental::row::rhs_index_type;

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
template <typename HaystackHasher, typename NeedleHasher>
struct hasher_adapter {
  hasher_adapter(HaystackHasher const& haystack_hasher, NeedleHasher const& needle_hasher)
    : _haystack_hasher{haystack_hasher}, _needle_hasher{needle_hasher}
  {
  }

  __device__ constexpr auto operator()(lhs_index_type idx) const noexcept
  {
    return _haystack_hasher(static_cast<size_type>(idx));
  }

  __device__ constexpr auto operator()(rhs_index_type idx) const noexcept
  {
    return _needle_hasher(static_cast<size_type>(idx));
  }

 private:
  HaystackHasher const _haystack_hasher;
  NeedleHasher const _needle_hasher;
};

/**
 * @brief An adapter functor to support strong index type for table row comparator that must be
 * operating on `cudf::size_type`.
 */
template <typename SelfComparator, typename TwoTableComparator>
struct comparator_adapter {
  comparator_adapter(SelfComparator const& self_comparator,
                     TwoTableComparator const& two_table_comparator)
    : _self_comparator{self_comparator}, _two_table_comparator{two_table_comparator}
  {
  }

  __device__ constexpr auto operator()(lhs_index_type lhs_index,
                                       lhs_index_type rhs_index) const noexcept
  {
    auto const lhs = static_cast<size_type>(lhs_index);
    auto const rhs = static_cast<size_type>(rhs_index);

    return _self_comparator(lhs, rhs);
  }

  __device__ constexpr auto operator()(lhs_index_type lhs_index,
                                       rhs_index_type rhs_index) const noexcept
  {
    return _two_table_comparator(lhs_index, rhs_index);
  }

 private:
  SelfComparator const _self_comparator;
  TwoTableComparator const _two_table_comparator;
};

/**
 * @brief Build a row bitmask for the input table.
 *
 * The output bitmask will have invalid bits corresponding to the input rows having nulls (at
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
    auto row_bitmask =
      cudf::detail::bitmask_and(
        table_view{nullable_columns}, stream, rmm::mr::get_current_device_resource())
        .first;
    auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
    return std::pair(std::move(row_bitmask), row_bitmask_ptr);
  }

  return std::pair(rmm::device_buffer{0, stream}, nullable_columns.front().null_mask());
}

}  // namespace
template <bool haystack_has_nulls, bool has_any_nulls, typename Func>
void dispatch(
  null_equality compare_nulls, auto self_comp, auto two_table_comp, auto nan_comp, Func func)
{
  auto const d_self_eq = self_comp.equal_to<haystack_has_nulls>(
    nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, nan_comp);
  auto const d_two_table_eq = two_table_comp.equal_to<has_any_nulls>(
    nullate::DYNAMIC{has_any_nulls}, compare_nulls, nan_comp);
  func(d_self_eq, d_two_table_eq);
}

/**
 * @brief Invoke an `operator()` template with a row equality comparator based on the specified
 * `compare_nans` parameter.
 *
 * @param compare_nans The flag to specify whether NaNs should be compared equal or not
 * @param func The input functor to invoke
 */
template <bool haystack_has_nulls, bool has_any_nulls, typename Func>
void dispatch_nan_comparator(nan_equality compare_nans,
                             null_equality compare_nulls,
                             auto self_comp,
                             auto two_table_comp,
                             Func func)
{
  if (compare_nans == nan_equality::ALL_EQUAL) {
    using nan_equal_comparator =
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
    dispatch<haystack_has_nulls, has_any_nulls>(
      compare_nulls, self_comp, two_table_comp, nan_equal_comparator{}, func);
  } else {
    using nan_unequal_comparator = cudf::experimental::row::equality::physical_equality_comparator;
    dispatch<haystack_has_nulls, has_any_nulls>(
      compare_nulls, self_comp, two_table_comp, nan_unequal_comparator{}, func);
  }
}

/**
 * @brief Check if rows in the given `needles` table exist in the `haystack` table.
 *
 * @param haystack The table containing the search space
 * @param needles A table of rows whose existence to check in the search space
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param compare_nans Control whether floating-point NaNs values should be compared as equal or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A vector of bools indicating if each row in `needles` has matching rows in `haystack`
 */
rmm::device_uvector<bool> contains(table_view const& haystack,
                                   table_view const& needles,
                                   null_equality compare_nulls,
                                   nan_equality compare_nans,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto const haystack_has_nulls = has_nested_nulls(haystack);
  auto const needles_has_nulls  = has_nested_nulls(needles);
  auto const has_any_nulls      = haystack_has_nulls || needles_has_nulls;

  auto const preprocessed_needles =
    cudf::experimental::row::equality::preprocessed_table::create(needles, stream);
  auto const preprocessed_haystack =
    cudf::experimental::row::equality::preprocessed_table::create(haystack, stream);

  auto const haystack_hasher   = cudf::experimental::row::hash::row_hasher(preprocessed_haystack);
  auto const d_haystack_hasher = haystack_hasher.device_hasher(nullate::DYNAMIC{has_any_nulls});
  auto const needle_hasher     = cudf::experimental::row::hash::row_hasher(preprocessed_needles);
  auto const d_needle_hasher   = needle_hasher.device_hasher(nullate::DYNAMIC{has_any_nulls});
  auto const d_hasher          = hasher_adapter{d_haystack_hasher, d_needle_hasher};
  using hasher_type            = decltype(d_hasher);

  auto const self_comparator =
    cudf::experimental::row::equality::self_comparator(preprocessed_haystack);
  auto const two_table_comparator = cudf::experimental::row::equality::two_table_comparator(
    preprocessed_haystack, preprocessed_needles);

  // The output vector.
  auto contained = rmm::device_uvector<bool>(needles.num_rows(), stream, mr);

  auto const haystack_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(auto idx) { return lhs_index_type{idx}; });
  auto const needles_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(auto idx) { return rhs_index_type{idx}; });

  auto const helper_func = [&](auto const& d_self_equal, auto const& d_two_table_equal) {
    auto const d_equal = comparator_adapter{d_self_equal, d_two_table_equal};

    auto set = cuco::experimental::static_set{
      cuco::experimental::extent{compute_hash_table_size(haystack.num_rows())},
      cuco::empty_key{lhs_index_type{-1}},
      d_equal,
      cuco::experimental::linear_probing<1, hasher_type>{d_hasher},
      detail::hash_table_allocator_type{default_allocator<lhs_index_type>{}, stream},
      stream.value()};

    if (haystack_has_nulls && compare_nulls == null_equality::UNEQUAL) {
      auto const row_bitmask = build_row_bitmask(haystack, stream).second;
      set.insert_if_async(haystack_iter,
                          haystack_iter + haystack.num_rows(),
                          thrust::counting_iterator<size_type>(0),  // stencil
                          row_is_valid{row_bitmask},
                          stream.value());
    } else {
      set.insert_async(haystack_iter, haystack_iter + haystack.num_rows(), stream.value());
    }

    if (needles_has_nulls && compare_nulls == null_equality::UNEQUAL) {
      set.contains_if_async(
        needles_iter, needles_iter + needles.num_rows(), contained.begin(), stream.value());
    } else {
      set.contains_async(
        needles_iter, needles_iter + needles.num_rows(), contained.begin(), stream.value());
    }
  };

  if (haystack_has_nulls) {
    if (has_any_nulls) {
      dispatch_nan_comparator<true, true>(
        compare_nans, compare_nulls, self_comparator, two_table_comparator, helper_func);
    }
  }

  return contained;
}

}  // namespace cudf::detail
