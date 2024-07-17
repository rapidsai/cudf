/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "join/join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/static_set.cuh>
#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>

#include <type_traits>

namespace cudf::detail {

namespace {

using cudf::experimental::row::lhs_index_type;
using cudf::experimental::row::rhs_index_type;

/**
 * @brief An hasher adapter wrapping both haystack hasher and needles hasher
 */
template <typename HaystackHasher, typename NeedleHasher>
struct hasher_adapter {
  hasher_adapter(HaystackHasher const& haystack_hasher, NeedleHasher const& needle_hasher)
    : _haystack_hasher{haystack_hasher}, _needle_hasher{needle_hasher}
  {
  }

  __device__ constexpr auto operator()(lhs_index_type idx) const noexcept
  {
    return _needle_hasher(static_cast<size_type>(idx));
  }

  __device__ constexpr auto operator()(rhs_index_type idx) const noexcept
  {
    return _haystack_hasher(static_cast<size_type>(idx));
  }

 private:
  HaystackHasher const _haystack_hasher;
  NeedleHasher const _needle_hasher;
};

/**
 * @brief An comparator adapter wrapping both self comparator and two table comparator
 */
template <typename SelfEqual, typename TwoTableEqual>
struct comparator_adapter {
  comparator_adapter(SelfEqual const& self_equal, TwoTableEqual const& two_table_equal)
    : _self_equal{self_equal}, _two_table_equal{two_table_equal}
  {
  }

  __device__ constexpr auto operator()(rhs_index_type lhs_index,
                                       rhs_index_type rhs_index) const noexcept
  {
    auto const lhs = static_cast<size_type>(lhs_index);
    auto const rhs = static_cast<size_type>(rhs_index);

    return _self_equal(lhs, rhs);
  }

  __device__ constexpr auto operator()(lhs_index_type lhs_index,
                                       rhs_index_type rhs_index) const noexcept
  {
    return _two_table_equal(lhs_index, rhs_index);
  }

 private:
  SelfEqual const _self_equal;
  TwoTableEqual const _two_table_equal;
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

/**
 * @brief Invokes the given `func` with desired comparators based on the specified `compare_nans`
 * parameter
 *
 * @tparam HasNested Flag indicating whether there are nested columns in haystack or needles
 * @tparam Hasher Type of device hash function
 * @tparam Func Type of the helper function doing `contains` check
 *
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param compare_nans Control whether floating-point NaNs values should be compared as equal or not
 * @param haystack_has_nulls Flag indicating whether haystack has nulls or not
 * @param has_any_nulls Flag indicating whether there are nested nulls is either haystack or needles
 * @param self_equal Self table comparator
 * @param two_table_equal Two table comparator
 * @param d_hasher Device hash functor
 * @param func The input functor to invoke
 */
template <bool HasNested, typename Hasher, typename Func>
void dispatch_nan_comparator(
  null_equality compare_nulls,
  nan_equality compare_nans,
  bool haystack_has_nulls,
  bool has_any_nulls,
  cudf::experimental::row::equality::self_comparator self_equal,
  cudf::experimental::row::equality::two_table_comparator two_table_equal,
  Hasher const& d_hasher,
  Func&& func)
{
  // Distinguish probing scheme CG sizes between nested and flat types for better performance
  auto const probing_scheme = [&]() {
    if constexpr (HasNested) {
      return cuco::linear_probing<4, Hasher>{d_hasher};
    } else {
      return cuco::linear_probing<1, Hasher>{d_hasher};
    }
  }();

  if (compare_nans == nan_equality::ALL_EQUAL) {
    using nan_equal_comparator =
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
    auto const d_self_equal = self_equal.equal_to<HasNested>(
      nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, nan_equal_comparator{});
    auto const d_two_table_equal = two_table_equal.equal_to<HasNested>(
      nullate::DYNAMIC{has_any_nulls}, compare_nulls, nan_equal_comparator{});
    func(d_self_equal, d_two_table_equal, probing_scheme);
  } else {
    using nan_unequal_comparator = cudf::experimental::row::equality::physical_equality_comparator;
    auto const d_self_equal      = self_equal.equal_to<HasNested>(
      nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, nan_unequal_comparator{});
    auto const d_two_table_equal = two_table_equal.equal_to<HasNested>(
      nullate::DYNAMIC{has_any_nulls}, compare_nulls, nan_unequal_comparator{});
    func(d_self_equal, d_two_table_equal, probing_scheme);
  }
}

}  // namespace

rmm::device_uvector<bool> contains(table_view const& haystack,
                                   table_view const& needles,
                                   null_equality compare_nulls,
                                   nan_equality compare_nans,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::have_same_types(haystack, needles), "Column types mismatch");

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

  auto const self_equal = cudf::experimental::row::equality::self_comparator(preprocessed_haystack);
  auto const two_table_equal = cudf::experimental::row::equality::two_table_comparator(
    preprocessed_needles, preprocessed_haystack);

  // The output vector.
  auto contained = rmm::device_uvector<bool>(needles.num_rows(), stream, mr);

  auto const haystack_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, cuda::proclaim_return_type<rhs_index_type>([] __device__(auto idx) {
      return rhs_index_type{idx};
    }));
  auto const needles_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, cuda::proclaim_return_type<lhs_index_type>([] __device__(auto idx) {
      return lhs_index_type{idx};
    }));

  auto const helper_func =
    [&](auto const& d_self_equal, auto const& d_two_table_equal, auto const& probing_scheme) {
      auto const d_equal = comparator_adapter{d_self_equal, d_two_table_equal};

      auto set = cuco::static_set{cuco::extent{compute_hash_table_size(haystack.num_rows())},
                                  cuco::empty_key{rhs_index_type{-1}},
                                  d_equal,
                                  probing_scheme,
                                  {},
                                  {},
                                  cudf::detail::cuco_allocator{stream},
                                  stream.value()};

      if (haystack_has_nulls && compare_nulls == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(haystack, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

        // If the haystack table has nulls but they are compared unequal, don't insert them.
        // Otherwise, it was known to cause performance issue:
        // - https://github.com/rapidsai/cudf/pull/6943
        // - https://github.com/rapidsai/cudf/pull/8277
        set.insert_if_async(haystack_iter,
                            haystack_iter + haystack.num_rows(),
                            thrust::counting_iterator<size_type>(0),  // stencil
                            row_is_valid{row_bitmask_ptr},
                            stream.value());
      } else {
        set.insert_async(haystack_iter, haystack_iter + haystack.num_rows(), stream.value());
      }

      if (needles_has_nulls && compare_nulls == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(needles, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
        set.contains_if_async(needles_iter,
                              needles_iter + needles.num_rows(),
                              thrust::counting_iterator<size_type>(0),  // stencil
                              row_is_valid{row_bitmask_ptr},
                              contained.begin(),
                              stream.value());
      } else {
        set.contains_async(
          needles_iter, needles_iter + needles.num_rows(), contained.begin(), stream.value());
      }
    };

  if (cudf::detail::has_nested_columns(haystack)) {
    dispatch_nan_comparator<true>(compare_nulls,
                                  compare_nans,
                                  haystack_has_nulls,
                                  has_any_nulls,
                                  self_equal,
                                  two_table_equal,
                                  d_hasher,
                                  helper_func);
  } else {
    dispatch_nan_comparator<false>(compare_nulls,
                                   compare_nans,
                                   haystack_has_nulls,
                                   has_any_nulls,
                                   self_equal,
                                   two_table_equal,
                                   d_hasher,
                                   helper_func);
  }

  return contained;
}

}  // namespace cudf::detail
