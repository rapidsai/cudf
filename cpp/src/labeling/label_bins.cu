/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/label_bins.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/labeling/label_bins.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/pair.h>

#include <limits>

namespace cudf {
namespace detail {
namespace {

// Sentinel used to indicate that an input value should be placed in the null
// bin.
// NOTE: In theory if a user decided to specify 2^31 bins this would fail. We
// could make this an error in Python, but that is such a crazy edge case...
constexpr size_type NULL_VALUE{std::numeric_limits<size_type>::max()};

/*
 * Functor for finding bins using thrust::transform.
 *
 * This functor is stateful, in the sense that it stores (for read-only use)
 * pointers to the edge ranges on construction to enable natural use with
 * thrust::transform semantics.  To handle null values, this functor assumes
 * that the input iterators have already been shifted to exclude the range
 * containing nulls. The `edge_index_shift` parameter is used to return the
 * index of a value's bin accounting for this shift.
 */
template <typename T,
          typename RandomAccessIterator,
          typename LeftComparator,
          typename RightComparator>
struct bin_finder {
  bin_finder(RandomAccessIterator left_begin,
             RandomAccessIterator left_end,
             RandomAccessIterator right_begin)
    : m_left_begin(left_begin), m_left_end(left_end), m_right_begin(right_begin)
  {
  }

  __device__ size_type operator()(thrust::pair<T, bool> input_value) const
  {
    // Immediately return sentinel for null inputs.
    if (!input_value.second) return NULL_VALUE;

    T value    = input_value.first;
    auto bound = thrust::lower_bound(thrust::seq, m_left_begin, m_left_end, value, m_left_comp);

    // Exit early and return sentinel for values that lie below the interval.
    if (bound == m_left_begin) { return NULL_VALUE; }

    auto index = thrust::distance(m_left_begin, thrust::prev(bound));
    return (m_right_comp(value, m_right_begin[index])) ? index : NULL_VALUE;
  }

  const RandomAccessIterator
    m_left_begin{};  // The beginning of the range containing the left bin edges.
  const RandomAccessIterator m_left_end{};  // The end of the range containing the left bin edges.
  const RandomAccessIterator
    m_right_begin{};                   // The beginning of the range containing the right bin edges.
  const LeftComparator m_left_comp{};  // Comparator used for left edges.
  const RightComparator m_right_comp{};  // Comparator used for right edges.
};

// Functor to identify rows that should be filtered out based on the sentinel set by
// bin_finder::operator().
struct filter_null_sentinel {
  __device__ bool operator()(size_type i) { return i != NULL_VALUE; }
};

// Bin the input by the edges in left_edges and right_edges.
template <typename T, typename LeftComparator, typename RightComparator>
std::unique_ptr<column> label_bins(column_view const& input,
                                   column_view const& left_edges,
                                   column_view const& right_edges,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto output = make_numeric_column(
    data_type(type_to_id<size_type>()), input.size(), mask_state::UNALLOCATED, stream, mr);
  auto output_mutable_view = output->mutable_view();
  auto output_begin        = output_mutable_view.begin<size_type>();
  auto output_end          = output_mutable_view.end<size_type>();

  // These device column views are necessary for creating iterators that work
  // for columns of compound types. The column_view iterators fail for compound
  // types because they return raw pointers to the start of the data. The output
  // does not require these iterators because it's always a primitive type.
  auto input_device_view       = column_device_view::create(input, stream);
  auto left_edges_device_view  = column_device_view::create(left_edges, stream);
  auto right_edges_device_view = column_device_view::create(right_edges, stream);

  auto left_begin  = left_edges_device_view->begin<T>();
  auto left_end    = left_edges_device_view->end<T>();
  auto right_begin = right_edges_device_view->begin<T>();

  using RandomAccessIterator = decltype(left_edges_device_view->begin<T>());

  if (input.has_nulls()) {
    thrust::transform(rmm::exec_policy(stream),
                      input_device_view->pair_begin<T, true>(),
                      input_device_view->pair_end<T, true>(),
                      output_begin,
                      bin_finder<T, RandomAccessIterator, LeftComparator, RightComparator>(
                        left_begin, left_end, right_begin));
  } else {
    thrust::transform(rmm::exec_policy(stream),
                      input_device_view->pair_begin<T, false>(),
                      input_device_view->pair_end<T, false>(),
                      output_begin,
                      bin_finder<T, RandomAccessIterator, LeftComparator, RightComparator>(
                        left_begin, left_end, right_begin));
  }

  const auto mask_and_count = valid_if(output_begin, output_end, filter_null_sentinel());

  output->set_null_mask(mask_and_count.first, mask_and_count.second);
  return output;
}

template <typename T>
constexpr auto is_supported_bin_type()
{
  return cudf::is_relationally_comparable<T, T>() && cudf::is_equality_comparable<T, T>();
}

struct bin_type_dispatcher {
  template <typename T, typename... Args>
  std::enable_if_t<not detail::is_supported_bin_type<T>(), std::unique_ptr<column>> operator()(
    Args&&...)
  {
    CUDF_FAIL("Type not support for cudf::bin");
  }

  template <typename T>
  std::enable_if_t<detail::is_supported_bin_type<T>(), std::unique_ptr<column>> operator()(
    column_view const& input,
    column_view const& left_edges,
    inclusive left_inclusive,
    column_view const& right_edges,
    inclusive right_inclusive,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::YES))
      return label_bins<T, thrust::less_equal<T>, thrust::less_equal<T>>(
        input, left_edges, right_edges, stream, mr);
    if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::NO))
      return label_bins<T, thrust::less_equal<T>, thrust::less<T>>(
        input, left_edges, right_edges, stream, mr);
    if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::YES))
      return label_bins<T, thrust::less<T>, thrust::less_equal<T>>(
        input, left_edges, right_edges, stream, mr);
    if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::NO))
      return label_bins<T, thrust::less<T>, thrust::less<T>>(
        input, left_edges, right_edges, stream, mr);

    CUDF_FAIL("Undefined inclusive setting.");
  }
};

}  // anonymous namespace

/// Bin the input by the edges in left_edges and right_edges.
std::unique_ptr<column> label_bins(column_view const& input,
                                   column_view const& left_edges,
                                   inclusive left_inclusive,
                                   column_view const& right_edges,
                                   inclusive right_inclusive,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE()
  CUDF_EXPECTS((input.type() == left_edges.type()) && (input.type() == right_edges.type()),
               "The input and edge columns must have the same types.");
  CUDF_EXPECTS(left_edges.size() == right_edges.size(),
               "The left and right edge columns must be of the same length.");
  CUDF_EXPECTS(!left_edges.has_nulls() && !right_edges.has_nulls(),
               "The left and right edge columns cannot contain nulls.");

  // Handle empty inputs.
  if (input.is_empty()) { return make_empty_column(data_type(type_to_id<size_type>())); }

  return type_dispatcher<dispatch_storage_type>(input.type(),
                                                detail::bin_type_dispatcher{},
                                                input,
                                                left_edges,
                                                left_inclusive,
                                                right_edges,
                                                right_inclusive,
                                                stream,
                                                mr);
}

}  // namespace detail

/// Bin the input by the edges in left_edges and right_edges.
std::unique_ptr<column> label_bins(column_view const& input,
                                   column_view const& left_edges,
                                   inclusive left_inclusive,
                                   column_view const& right_edges,
                                   inclusive right_inclusive,
                                   rmm::mr::device_memory_resource* mr)
{
  return detail::label_bins(
    input, left_edges, left_inclusive, right_edges, right_inclusive, rmm::cuda_stream_default, mr);
}
}  // namespace cudf
