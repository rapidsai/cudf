/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/search.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <thrust/binary_search.h>
#include <hash/unordered_multiset.cuh>

namespace cudf {
namespace {
template <typename DataIterator,
          typename ValuesIterator,
          typename OutputIterator,
          typename Comparator>
void launch_search(DataIterator it_data,
                   ValuesIterator it_vals,
                   size_type data_size,
                   size_type values_size,
                   OutputIterator it_output,
                   Comparator comp,
                   bool find_first,
                   cudaStream_t stream)
{
  if (find_first) {
    thrust::lower_bound(rmm::exec_policy(stream)->on(stream),
                        it_data,
                        it_data + data_size,
                        it_vals,
                        it_vals + values_size,
                        it_output,
                        comp);
  } else {
    thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
                        it_data,
                        it_data + data_size,
                        it_vals,
                        it_vals + values_size,
                        it_output,
                        comp);
  }
}

std::unique_ptr<column> search_ordered(table_view const& t,
                                       table_view const& values,
                                       bool find_first,
                                       std::vector<order> const& column_order,
                                       std::vector<null_order> const& null_precedence,
                                       rmm::mr::device_memory_resource* mr,
                                       cudaStream_t stream = 0)
{
  // Allocate result column
  std::unique_ptr<column> result = make_numeric_column(
    data_type{type_to_id<size_type>()}, values.num_rows(), mask_state::UNALLOCATED, stream, mr);

  mutable_column_view result_view = result.get()->mutable_view();

  // Handle empty inputs
  if (t.num_rows() == 0) {
    CUDA_TRY(cudaMemset(result_view.data<size_type>(), 0, values.num_rows() * sizeof(size_type)));
    return result;
  }

  if (not column_order.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(t.num_columns()) == column_order.size(),
                 "Mismatch between number of columns and column order.");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(t.num_columns()) == null_precedence.size(),
                 "Mismatch between number of columns and null precedence.");
  }

  auto d_t      = table_device_view::create(t, stream);
  auto d_values = table_device_view::create(values, stream);
  auto count_it = thrust::make_counting_iterator<size_type>(0);

  rmm::device_vector<order> d_column_order(column_order.begin(), column_order.end());
  rmm::device_vector<null_order> d_null_precedence(null_precedence.begin(), null_precedence.end());

  if (has_nulls(t) or has_nulls(values)) {
    auto ineq_op =
      (find_first)
        ? row_lexicographic_comparator<true>(
            *d_t, *d_values, d_column_order.data().get(), d_null_precedence.data().get())
        : row_lexicographic_comparator<true>(
            *d_values, *d_t, d_column_order.data().get(), d_null_precedence.data().get());

    launch_search(count_it,
                  count_it,
                  t.num_rows(),
                  values.num_rows(),
                  result_view.data<size_type>(),
                  ineq_op,
                  find_first,
                  stream);
  } else {
    auto ineq_op =
      (find_first)
        ? row_lexicographic_comparator<false>(
            *d_t, *d_values, d_column_order.data().get(), d_null_precedence.data().get())
        : row_lexicographic_comparator<false>(
            *d_values, *d_t, d_column_order.data().get(), d_null_precedence.data().get());

    launch_search(count_it,
                  count_it,
                  t.num_rows(),
                  values.num_rows(),
                  result_view.data<size_type>(),
                  ineq_op,
                  find_first,
                  stream);
  }

  return result;
}

struct contains_scalar_dispatch {
  template <typename Element>
  bool operator()(column_view const& col, scalar const& value, cudaStream_t stream)
  {
    CUDF_EXPECTS(col.type() == value.type(), "scalar and column types must match");

    using Type       = device_storage_type_t<Element>;
    using ScalarType = cudf::scalar_type_t<Element>;
    auto d_col       = column_device_view::create(col, stream);
    auto s           = static_cast<const ScalarType*>(&value);

    if (col.has_nulls()) {
      auto found_iter = thrust::find(rmm::exec_policy(stream)->on(stream),
                                     d_col->pair_begin<Type, true>(),
                                     d_col->pair_end<Type, true>(),
                                     thrust::make_pair(s->value(), true));

      return found_iter != d_col->pair_end<Type, true>();
    } else {
      auto found_iter = thrust::find(rmm::exec_policy(stream)->on(stream),  //
                                     d_col->begin<Type>(),
                                     d_col->end<Type>(),
                                     s->value());

      return found_iter != d_col->end<Type>();
    }
  }
};

template <>
bool contains_scalar_dispatch::operator()<cudf::list_view>(column_view const& col,
                                                           scalar const& value,
                                                           cudaStream_t stream)
{
  CUDF_FAIL("list_view type not supported yet");
}

template <>
bool contains_scalar_dispatch::operator()<cudf::struct_view>(column_view const& col,
                                                             scalar const& value,
                                                             cudaStream_t stream)
{
  CUDF_FAIL("struct_view type not supported yet");
}

template <>
bool contains_scalar_dispatch::operator()<cudf::dictionary32>(column_view const& col,
                                                              scalar const& value,
                                                              cudaStream_t stream)
{
  auto dict_col = cudf::dictionary_column_view(col);
  // first, find the value in the dictionary's key set
  auto index = cudf::dictionary::detail::get_index(
    dict_col, value, rmm::mr::get_current_device_resource(), stream);
  // if found, check the index is actually in the indices column
  return index->is_valid() ? cudf::type_dispatcher(dict_col.indices().type(),
                                                   contains_scalar_dispatch{},
                                                   dict_col.indices(),
                                                   *index,
                                                   stream)
                           : false;
}

}  // namespace

namespace detail {
bool contains(column_view const& col, scalar const& value, cudaStream_t stream)
{
  if (col.size() == 0) { return false; }

  if (not value.is_valid()) { return col.has_nulls(); }

  return cudf::type_dispatcher(col.type(), contains_scalar_dispatch{}, col, value, stream);
}

struct multi_contains_dispatch {
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    std::unique_ptr<column> result = make_numeric_column(data_type{type_to_id<bool>()},
                                                         haystack.size(),
                                                         copy_bitmask(haystack),
                                                         haystack.null_count(),
                                                         stream,
                                                         mr);

    if (haystack.size() == 0) { return result; }

    mutable_column_view result_view = result.get()->mutable_view();

    if (needles.size() == 0) {
      thrust::fill(rmm::exec_policy(stream)->on(stream),
                   result_view.begin<bool>(),
                   result_view.end<bool>(),
                   false);
      return result;
    }

    auto hash_set        = cudf::detail::unordered_multiset<Element>::create(needles, stream);
    auto device_hash_set = hash_set.to_device();

    auto d_haystack_ptr = column_device_view::create(haystack, stream);
    auto d_haystack     = *d_haystack_ptr;

    if (haystack.has_nulls()) {
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(haystack.size()),
                        result_view.begin<bool>(),
                        [device_hash_set, d_haystack] __device__(size_t index) {
                          return d_haystack.is_null_nocheck(index) ||
                                 device_hash_set.contains(d_haystack.element<Element>(index));
                        });
    } else {
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(haystack.size()),
                        result_view.begin<bool>(),
                        [device_hash_set, d_haystack] __device__(size_t index) {
                          return device_hash_set.contains(d_haystack.element<Element>(index));
                        });
    }

    return result;
  }
};

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<list_view>(
  column_view const& haystack,
  column_view const& needles,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  CUDF_FAIL("list_view type not supported");
}

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<struct_view>(
  column_view const& haystack,
  column_view const& needles,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  CUDF_FAIL("struct_view type not supported");
}

template <>
std::unique_ptr<column> multi_contains_dispatch::operator()<dictionary32>(
  column_view const& haystack_in,
  column_view const& needles_in,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  dictionary_column_view const haystack(haystack_in);
  dictionary_column_view const needles(needles_in);
  // first combine keys so both dictionaries have the same set
  auto haystack_matched = dictionary::detail::add_keys(
    haystack, needles.keys(), rmm::mr::get_current_device_resource(), stream);
  auto const haystack_view = dictionary_column_view(haystack_matched->view());
  auto needles_matched     = dictionary::detail::set_keys(
    needles, haystack_view.keys(), rmm::mr::get_current_device_resource(), stream);
  auto const needles_view = dictionary_column_view(needles_matched->view());

  // now just use the indices for the contains
  column_view const haystack_indices = haystack_view.get_indices_annotated();
  column_view const needles_indices  = needles_view.get_indices_annotated();
  return cudf::type_dispatcher(haystack_indices.type(),
                               multi_contains_dispatch{},
                               haystack_indices,
                               needles_indices,
                               mr,
                               stream);
}

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::mr::device_memory_resource* mr,
                                 cudaStream_t stream)
{
  CUDF_EXPECTS(haystack.type() == needles.type(), "DTYPE mismatch");

  return cudf::type_dispatcher(
    haystack.type(), multi_contains_dispatch{}, haystack, needles, mr, stream);
}

std::unique_ptr<column> lower_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream)
{
  return search_ordered(t, values, true, column_order, null_precedence, mr, stream);
}

std::unique_ptr<column> upper_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream)
{
  return search_ordered(t, values, false, column_order, null_precedence, mr, stream);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> lower_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::lower_bound(t, values, column_order, null_precedence, mr);
}

std::unique_ptr<column> upper_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::upper_bound(t, values, column_order, null_precedence, mr);
}

bool contains(column_view const& col, scalar const& value)
{
  CUDF_FUNC_RANGE();
  return detail::contains(col, value);
}

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(haystack, needles, mr);
}

}  // namespace cudf
