/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/search.hpp>
#include <cudf/legacy/copying.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/scalar/scalar_device_view.cuh>

#include <hash/unordered_multiset.cuh>
#include <cudf/detail/iterator.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <strings/utilities.hpp>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>

namespace cudf {
namespace experimental {

namespace {

template <typename DataIterator, typename ValuesIterator,
          typename OutputIterator, typename Comparator>
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
                        it_data, it_data + data_size,
                        it_vals, it_vals + values_size,
                        it_output, comp);
  }
  else {
    thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
                        it_data, it_data + data_size,
                        it_vals, it_vals + values_size,
                        it_output, comp);
  }
}

} // namespace

namespace detail {

std::unique_ptr<column> search_ordered(table_view const& t,
                                       table_view const& values,
                                       bool find_first,
                                       std::vector<order> const& column_order,
                                       std::vector<null_order> const& null_precedence,
                                       rmm::mr::device_memory_resource *mr,
                                       cudaStream_t stream = 0)
{
  // Allocate result column
  std::unique_ptr<column> result = make_numeric_column(data_type{experimental::type_to_id<size_type>()}, values.num_rows(),
                                                       mask_state::UNALLOCATED, stream, mr);

  mutable_column_view result_view = result.get()->mutable_view();

  // Handle empty inputs
  if (t.num_rows() == 0) {
    CUDA_TRY(cudaMemset(result_view.data<size_type>(), 0, values.num_rows() * sizeof(size_type)));
    return result;
  }

  if (not column_order.empty()) {
    CUDF_EXPECTS(
        static_cast<std::size_t>(t.num_columns()) == column_order.size(),
        "Mismatch between number of columns and column order.");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(
        static_cast<std::size_t>(t.num_columns()) == null_precedence.size(),
        "Mismatch between number of columns and null precedence.");
  }

  auto d_t      = table_device_view::create(t, stream);
  auto d_values = table_device_view::create(values, stream);
  auto count_it = thrust::make_counting_iterator<size_type>(0);

  rmm::device_vector<order> d_column_order(column_order.begin(), column_order.end());
  rmm::device_vector<null_order> d_null_precedence(null_precedence.begin(), null_precedence.end());

  if (has_nulls(t)) {
    auto ineq_op = (find_first)
      ? row_lexicographic_comparator<true>(*d_t, *d_values, d_column_order.data().get(), d_null_precedence.data().get())
      : row_lexicographic_comparator<true>(*d_values, *d_t, d_column_order.data().get(), d_null_precedence.data().get());

    launch_search(count_it, count_it, t.num_rows(), values.num_rows(),
                  result_view.data<size_type>(), ineq_op, find_first, stream);
  } else {
    auto ineq_op = (find_first)
      ? row_lexicographic_comparator<false>(*d_t, *d_values, d_column_order.data().get(), d_null_precedence.data().get())
      : row_lexicographic_comparator<false>(*d_values, *d_t, d_column_order.data().get(), d_null_precedence.data().get());

    launch_search(count_it, count_it, t.num_rows(), values.num_rows(),
                  result_view.data<size_type>(), ineq_op, find_first, stream);
  }

  return result;
}

struct contains_scalar_dispatch {
  template <typename Element>
  bool operator()(column_view const& col, scalar const& value,
                  cudaStream_t stream, rmm::mr::device_memory_resource *mr) {

    using ScalarType = cudf::experimental::scalar_type_t<Element>;
    auto d_col = column_device_view::create(col, stream);
    auto s = static_cast<const ScalarType *>(&value);

    if (col.has_nulls()) {
      auto found_iter = thrust::find(rmm::exec_policy(stream)->on(stream),
                                     d_col->pair_begin<Element, true>(),
                                     d_col->pair_end<Element, true>(),
                                     thrust::make_pair(s->value(), true));

      return found_iter != d_col->pair_end<Element, true>();
    } else {
      auto found_iter =  thrust::find(rmm::exec_policy(stream)->on(stream),
                                      d_col->begin<Element>(),
                                      d_col->end<Element>(), s->value());

      return found_iter != d_col->end<Element>();
    }
  }
};

bool contains(column_view const& col,
              scalar const& value,
              rmm::mr::device_memory_resource *mr,
              cudaStream_t stream = 0)
{
  CUDF_EXPECTS(col.type() == value.type(), "DTYPE mismatch");

  if (col.size() == 0) {
    return false;
  }

  if (not value.is_valid()) {
    return col.has_nulls();
  }

  return cudf::experimental::type_dispatcher(col.type(),
                                             contains_scalar_dispatch{},
                                             col, value,
                                             stream, mr);
}

struct multi_contains_dispatch {
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& haystack,
                                     column_view const& needles,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream) {

    std::unique_ptr<column> result = make_numeric_column(data_type{experimental::type_to_id<bool8>()},
                                                         haystack.size(),
                                                         copy_bitmask(haystack),
                                                         haystack.null_count(),
                                                         stream, mr);
  
    if (haystack.size() == 0) {
      return result;
    }

    mutable_column_view result_view = result.get()->mutable_view();

    if (needles.size() == 0) {
      bool8 f = false;
      thrust::fill(rmm::exec_policy(stream)->on(stream), result_view.begin<bool8>(), result_view.end<bool8>(), f);
      return result;
    }

    auto hash_set = cudf::detail::unordered_multiset<Element>::create(needles, stream);
    auto device_hash_set = hash_set.to_device();

    auto d_haystack_ptr = column_device_view::create(haystack, stream);
    auto d_haystack = *d_haystack_ptr;

    if (haystack.has_nulls()) {
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(haystack.size()),
                        result_view.begin<bool8>(),
                        [device_hash_set, d_haystack] __device__ (size_t index) {
                          return d_haystack.is_null_nocheck(index) || device_hash_set.contains(d_haystack.element<Element>(index));
                        });
    } else {
      thrust::transform(rmm::exec_policy(stream)->on(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(haystack.size()),
                        result_view.begin<bool8>(),
                        [device_hash_set, d_haystack] __device__ (size_t index) {
                          return device_hash_set.contains(d_haystack.element<Element>(index));
                        });
    }

    return result;
  }
};

std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 rmm::mr::device_memory_resource* mr,
                                 cudaStream_t stream = 0) {

  CUDF_EXPECTS(haystack.type() == needles.type(), "DTYPE mismatch");

  return cudf::experimental::type_dispatcher(haystack.type(),
                                             multi_contains_dispatch{},
                                             haystack, needles, mr, stream);
}
} // namespace detail

std::unique_ptr<column> lower_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource *mr)
{
  return detail::search_ordered(t, values, true, column_order, null_precedence, mr);
}

std::unique_ptr<column> upper_bound(table_view const& t,
                                    table_view const& values,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    rmm::mr::device_memory_resource *mr)
{
  return detail::search_ordered(t, values, false, column_order, null_precedence, mr);
}

bool contains(column_view const& col, scalar const& value, rmm::mr::device_memory_resource *mr)
{
  return detail::contains(col, value, mr);
}

std::unique_ptr<column> contains(column_view const& haystack, column_view const& needles,
                                       rmm::mr::device_memory_resource* mr) {
  return detail::contains(haystack, needles, mr);
}

} // namespace exp
} // namespace cudf
