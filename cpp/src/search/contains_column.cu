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

#include <hash/unordered_multiset.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Get the underlying value of a scalar through a scalar device view.
 *
 * @tparam Type The scalar's value type
 * @tparam ScalarDView Type of the input scalar device view
 * @param d_scalar The input scalar device view
 */
template <typename Type, typename ScalarDView>
__device__ auto inline get_scalar_value(ScalarDView d_scalar)
{
  if constexpr (cudf::is_fixed_point<Type>()) {
    return d_scalar.rep();
  } else {
    return d_scalar.value();
  }
}

struct contains_scalar_dispatch {
  // SFINAE with conditional return type because we need to support device lambda in this function.
  // This is required due to a limitation of nvcc.
  template <typename Type>
  std::enable_if_t<!is_nested<Type>(), bool> operator()(column_view const& haystack,
                                                        scalar const& needle,
                                                        rmm::cuda_stream_view stream) const
  {
    CUDF_EXPECTS(haystack.type() == needle.type(), "Scalar and column types must match");

    // If the input scalar is invalid, it should be handled by the caller before
    // dispatching to this function.
    // (Handling such case is very simple: just check if the input haystack column has nulls).

    using DType           = device_storage_type_t<Type>;
    auto const d_haystack = column_device_view::create(haystack, stream);
    auto const d_needle =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<Type>&>(const_cast<scalar&>(needle)));

    if (haystack.has_nulls()) {
      auto const begin = d_haystack->pair_begin<DType, true>();
      auto const end   = d_haystack->pair_end<DType, true>();

      return thrust::count_if(
               rmm::exec_policy(stream), begin, end, [d_needle] __device__(auto const val_pair) {
                 auto const needle_pair = thrust::make_pair(get_scalar_value<Type>(d_needle), true);
                 return val_pair == needle_pair;
               }) > 0;
    } else {
      auto const begin = d_haystack->begin<DType>();
      auto const end   = d_haystack->end<DType>();

      return thrust::count_if(
               rmm::exec_policy(stream), begin, end, [d_needle] __device__(auto const val) {
                 return val == get_scalar_value<Type>(d_needle);
               }) > 0;
    }
  }

  template <typename Type>
  std::enable_if_t<is_nested<Type>(), bool> operator()(column_view const& haystack,
                                                       scalar const& needle,
                                                       rmm::cuda_stream_view stream) const
  {
    CUDF_EXPECTS(haystack.type() == needle.type(), "Scalar and column types must match");
    // Haystack and needle structure compatibility will be checked by the table comparator
    // constructor during call to `contains_nested_element`.

    // If the input scalar is invalid, it should be handled by the caller before
    // dispatching to this function.
    // (Handling such case is very simple: just check if the input haystack column has nulls).

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

    if (haystack.has_nulls()) {
      auto const haystack_cdv_ptr  = column_device_view::create(haystack, stream);
      auto const haystack_valid_it = cudf::detail::make_validity_iterator<false>(*haystack_cdv_ptr);

      return thrust::count_if(rmm::exec_policy(stream),
                              begin,
                              end,
                              [d_comp, haystack_valid_it] __device__(auto const idx) {
                                if (!haystack_valid_it[static_cast<size_type>(idx)]) {
                                  return false;
                                }
                                return d_comp(
                                  idx, rhs_index_type{0});  // compare haystack[idx] == needle[0].
                              }) > 0;
    }

    return thrust::count_if(
             rmm::exec_policy(stream), begin, end, [d_comp] __device__(auto const idx) {
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
  auto const index = cudf::dictionary::detail::get_index(dict_col, needle, stream);
  // if found, check the index is actually in the indices column
  return index->is_valid(stream) && cudf::type_dispatcher(dict_col.indices().type(),
                                                          contains_scalar_dispatch{},
                                                          dict_col.indices(),
                                                          *index,
                                                          stream);
}

struct contains_column_dispatch {
  template <typename ElementType, typename Haystack>
  struct contains_fn {
    bool __device__ operator()(size_type const idx) const
    {
      if (needles_have_nulls && needles.is_null_nocheck(idx)) {
        // Exit early. The value doesn't matter, and will be masked as a null element.
        return true;
      }

      return haystack.contains(needles.template element<ElementType>(idx));
    }

    Haystack const haystack;
    column_device_view const needles;
    bool const needles_have_nulls;
  };

  template <typename Type, CUDF_ENABLE_IF(!is_nested<Type>())>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto result = make_numeric_column(data_type{type_to_id<bool>()},
                                      needles.size(),
                                      copy_bitmask(needles, stream, mr),
                                      needles.null_count(),
                                      stream,
                                      mr);
    if (needles.is_empty()) { return result; }

    auto const out_begin = result->mutable_view().template begin<bool>();
    if (haystack.is_empty()) {
      thrust::uninitialized_fill(
        rmm::exec_policy(stream), out_begin, out_begin + needles.size(), false);
      return result;
    }

    auto const haystack_set    = cudf::detail::unordered_multiset<Type>::create(haystack, stream);
    auto const haystack_set_dv = haystack_set.to_device();
    auto const needles_cdv_ptr = column_device_view::create(needles, stream);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(needles.size()),
                      out_begin,
                      contains_fn<Type, decltype(haystack_set_dv)>{
                        haystack_set_dv, *needles_cdv_ptr, needles.has_nulls()});
    return result;
  }

  template <typename Type, CUDF_ENABLE_IF(is_nested<Type>())>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto result_v = contains(table_view{{haystack}},
                             table_view{{needles}},
                             null_equality::EQUAL,
                             nan_equality::ALL_EQUAL,
                             stream,
                             mr);
    return std::make_unique<column>(
      std::move(result_v), copy_bitmask(needles, stream, mr), needles.null_count());
  }
};

template <>
std::unique_ptr<column> contains_column_dispatch::operator()<dictionary32>(
  column_view const& haystack_in,
  column_view const& needles_in,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) const
{
  dictionary_column_view const haystack(haystack_in);
  dictionary_column_view const needles(needles_in);
  // first combine keys so both dictionaries have the same set
  auto needles_matched     = dictionary::detail::add_keys(needles, haystack.keys(), stream);
  auto const needles_view  = dictionary_column_view(needles_matched->view());
  auto haystack_matched    = dictionary::detail::set_keys(haystack, needles_view.keys(), stream);
  auto const haystack_view = dictionary_column_view(haystack_matched->view());

  // now just use the indices for the contains
  column_view const haystack_indices = haystack_view.get_indices_annotated();
  column_view const needles_indices  = needles_view.get_indices_annotated();
  return cudf::type_dispatcher(haystack_indices.type(),
                               contains_column_dispatch{},
                               haystack_indices,
                               needles_indices,
                               stream,
                               mr);
}
}  // namespace

bool contains(column_view const& haystack, scalar const& needle, rmm::cuda_stream_view stream)
{
  if (haystack.is_empty()) { return false; }
  if (not needle.is_valid(stream)) { return haystack.has_nulls(); }

  return cudf::type_dispatcher(
    haystack.type(), contains_scalar_dispatch{}, haystack, needle, stream);
}

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(haystack.type() == needles.type(), "DTYPE mismatch");

  return cudf::type_dispatcher(
    haystack.type(), contains_column_dispatch{}, haystack, needles, stream, mr);
}

}  // namespace detail

bool contains(column_view const& haystack, scalar const& needle)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needle, cudf::default_stream_value);
}

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needles, cudf::default_stream_value, mr);
}

}  // namespace cudf
