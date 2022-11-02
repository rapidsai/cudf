/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/pair.h>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Get the underlying value of a scalar through a scalar device view.
 *
 * @tparam Element The scalar's value type
 * @tparam ScalarDView Type of the input scalar device view
 * @param d_scalar The input scalar device view
 */
template <typename Element, typename ScalarDView>
__device__ auto inline get_scalar_value(ScalarDView d_scalar)
{
  if constexpr (cudf::is_fixed_point<Element>()) {
    return d_scalar.rep();
  } else {
    return d_scalar.value();
  }
}

struct contains_scalar_dispatch {
  // SFINAE with conditional return type because we need to support device lambda in this function.
  // This is required due to a limitation of nvcc.
  template <typename Element>
  std::enable_if_t<!is_nested<Element>(), bool> operator()(column_view const& haystack,
                                                           scalar const& needle,
                                                           rmm::cuda_stream_view stream) const
  {
    CUDF_EXPECTS(haystack.type() == needle.type(), "Scalar and column types must match");
    // Don't need to check for needle validity. If it is invalid, it should be handled by the caller
    // before dispatching to this function.

    using DType           = device_storage_type_t<Element>;
    auto const d_haystack = column_device_view::create(haystack, stream);
    auto const d_needle   = get_scalar_device_view(
      static_cast<cudf::scalar_type_t<Element>&>(const_cast<scalar&>(needle)));

    auto const begin =
      d_haystack->optional_begin<DType>(cudf::nullate::DYNAMIC{haystack.has_nulls()});
    auto const end = d_haystack->optional_end<DType>(cudf::nullate::DYNAMIC{haystack.has_nulls()});

    return thrust::count_if(
             rmm::exec_policy(stream), begin, end, [d_needle] __device__(auto const val_pair) {
               auto needle = get_scalar_value<Element>(d_needle);
               return val_pair.has_value() && (needle == *val_pair);
             }) > 0;
  }

  template <typename Element>
  std::enable_if_t<is_nested<Element>(), bool> operator()(column_view const& haystack,
                                                          scalar const& needle,
                                                          rmm::cuda_stream_view stream) const
  {
    CUDF_EXPECTS(haystack.type() == needle.type(), "Scalar and column types must match");
    // Don't need to check for needle validity. If it is invalid, it should be handled by the caller
    // before dispatching to this function.
    // In addition, haystack and needle structure compatibility will be checked later on by
    // constructor of the table comparator.

    auto const haystack_tv   = table_view{{haystack}};
    auto const needle_as_col = make_column_from_scalar(needle, 1, stream);
    auto const needle_tv     = table_view{{needle_as_col->view()}};
    auto const has_nulls     = has_nested_nulls(haystack_tv) || has_nested_nulls(needle_tv);

    auto const comparator =
      cudf::experimental::row::equality::two_table_comparator(haystack_tv, needle_tv, stream);
    auto const d_comp = comparator.equal_to(nullate::DYNAMIC{has_nulls});

    auto const begin = cudf::experimental::row::lhs_iterator(0);
    auto const end   = begin + haystack.size();
    using cudf::experimental::row::rhs_index_type;

    auto const check_nulls      = haystack.has_nulls();
    auto const haystack_cdv_ptr = column_device_view::create(haystack, stream);

    return thrust::count_if(
             rmm::exec_policy(stream),
             begin,
             end,
             [d_comp, check_nulls, d_haystack = *haystack_cdv_ptr] __device__(auto const idx) {
               if (check_nulls && d_haystack.is_null_nocheck(static_cast<size_type>(idx))) {
                 return false;
               }
               return d_comp(idx, rhs_index_type{0});  // compare haystack[idx] == needle[0].
             }) > 0;
  }
};

template <>
bool contains_scalar_dispatch::operator()<cudf::dictionary32>(column_view const& haystack,
                                                              scalar const& needle,
                                                              rmm::cuda_stream_view stream) const
{
  auto const dict_col = cudf::dictionary_column_view(haystack);
  // first, find the needle in the dictionary's key set
  auto const index = cudf::dictionary::detail::get_index(
    dict_col, needle, stream, rmm::mr::get_current_device_resource());
  // if found, check the index is actually in the indices column
  return index->is_valid(stream) && cudf::type_dispatcher(dict_col.indices().type(),
                                                          contains_scalar_dispatch{},
                                                          dict_col.indices(),
                                                          *index,
                                                          stream);
}

}  // namespace

bool contains(column_view const& haystack, scalar const& needle, rmm::cuda_stream_view stream)
{
  if (haystack.is_empty()) { return false; }
  if (not needle.is_valid(stream)) { return haystack.has_nulls(); }

  return cudf::type_dispatcher(
    haystack.type(), contains_scalar_dispatch{}, haystack, needle, stream);
}

}  // namespace detail

bool contains(column_view const& haystack, scalar const& needle)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needle, cudf::get_default_stream());
}

}  // namespace cudf
