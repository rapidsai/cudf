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

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/advance.h>
#include <thrust/pair.h>
#include <cudf/binning/bin.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <limits>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf {

// Sentinel used to indicate that an input value should be placed in the null
// bin.
// NOTE: In theory if a user decided to specify 2^31 bins this would fail. We
// could make this an error in Python, but that is such a crazy edge case...
constexpr size_type NULL_VALUE{std::numeric_limits<size_type>::max()};

namespace detail {
namespace {

template <typename T, typename RandomAccessIterator, typename LeftComparator, typename RightComparator>
struct bin_finder {
  bin_finder(RandomAccessIterator left_begin, RandomAccessIterator left_end, RandomAccessIterator right_begin)
    : m_left_begin(left_begin), m_left_end(left_end), m_right_begin(right_begin)
  {
  }

  __device__ size_type operator()(thrust::pair<T, bool> input_value) const
  {
    // Immediately return sentinel for null inputs.
    if (!input_value.second) return NULL_VALUE;

    T value = input_value.first;
    auto bound =
      thrust::lower_bound(thrust::seq, m_left_begin, m_left_end, value, m_left_comp);

    // Exit early and return sentinel for values that lie below the interval.
    if (bound == m_left_begin) { return NULL_VALUE; }

    auto index = thrust::distance(m_left_begin, thrust::prev(bound));
    return (m_right_comp(value, m_right_begin[index])) ? index : NULL_VALUE;
  }

  RandomAccessIterator m_left_begin{};   // The beginning of the range containing the left bin edges.
  RandomAccessIterator m_left_end{};     // The end of the range containing the left bin edges.
  RandomAccessIterator m_right_begin{};  // The beginning of the range containing the right bin edges.
  LeftComparator m_left_comp{};          // Comparator used for left edges.
  RightComparator m_right_comp{};        // Comparator used for right edges.
};

// Functor to identify rows that should be filtered out based on the sentinel set by
// bin_finder::operator().
struct filter_null_sentinel {
  __device__ bool operator()(size_type i) { return i != NULL_VALUE; }
};

/// Bin the input by the edges in left_edges and right_edges.
template <typename T, typename LeftComparator, typename RightComparator, bool InputIsNullable>
std::unique_ptr<column> bin(column_view const& input,
                            column_view const& left_edges,
                            column_view const& right_edges,
                            rmm::mr::device_memory_resource* mr)
{
  auto output = cudf::make_numeric_column(data_type(type_to_id<size_type>()), input.size());
  auto output_mutable_view = output->mutable_view();
  auto input_device_view   = column_device_view::create(input);

  thrust::transform(thrust::device,
                    input_device_view->pair_begin<T, InputIsNullable>(),
                    input_device_view->pair_end<T, InputIsNullable>(),
                    output_mutable_view.begin<size_type>(),
                    // Must specify const T as the template type because the column
                    // views provided on the edges will always return const types. The
                    // template arguments to `datq` need not specify const since that
                    // const is added as part of the signature.
                    bin_finder<T, decltype(left_edges.begin<T>()), LeftComparator, RightComparator>(
                      left_edges.begin<T>(), left_edges.end<T>(),
                      right_edges.begin<T>()));

  auto mask_and_count = cudf::detail::valid_if(output_mutable_view.begin<size_type>(),
                                               output_mutable_view.end<size_type>(),
                                               filter_null_sentinel());

  output->set_null_mask(mask_and_count.first, mask_and_count.second);
  return output;
}

template <typename T>
constexpr auto is_supported_bin_type()
{
  // TODO: Determine what other types (such as fixed point numbers) should be
  // supported, and whether any of them (like strings) require special
  // handling.
  return (cudf::is_numeric<T>() && !std::is_same<T, bool>::value);
}

}  // anonymous namespace
}  // namespace detail

/// Functor suitable for use with type_dispatcher that exploits SFINAE to call the appropriate
/// detail::bin method.
struct bin_type_dispatcher {
  template <typename T, typename... Args>
  std::enable_if_t<not detail::is_supported_bin_type<T>(), std::unique_ptr<column>> operator()(
    Args&&... args)
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
    rmm::mr::device_memory_resource* mr)
  {
    // Note: We could be slightly more efficient in the not nullable case
    // by overloading the call operator of bin_finder to accept a value in
    // addition to a pair and using a raw (non-pair) iterator in the
    // transform call in detail::bin, if we need to make this faster.
    if (input.nullable()) {
      // Using a switch statement might be more appropriate for an enum, but it's far more verbose
      // in this case.
      if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::YES))
        return detail::bin<T, thrust::less_equal<T>, thrust::less_equal<T>, true>(
          input, left_edges, right_edges, mr);
      if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::NO))
        return detail::bin<T, thrust::less_equal<T>, thrust::less<T>, true>(
          input, left_edges, right_edges, mr);
      if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::YES))
        return detail::bin<T, thrust::less<T>, thrust::less_equal<T>, true>(
          input, left_edges, right_edges, mr);
      if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::NO))
        return detail::bin<T, thrust::less<T>, thrust::less<T>, true>(
          input, left_edges, right_edges, mr);
    } else {
      if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::YES))
        return detail::bin<T, thrust::less_equal<T>, thrust::less_equal<T>, false>(
          input, left_edges, right_edges, mr);
      if ((left_inclusive == inclusive::YES) && (right_inclusive == inclusive::NO))
        return detail::bin<T, thrust::less_equal<T>, thrust::less<T>, false>(
          input, left_edges, right_edges, mr);
      if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::YES))
        return detail::bin<T, thrust::less<T>, thrust::less_equal<T>, false>(
          input, left_edges, right_edges, mr);
      if ((left_inclusive == inclusive::NO) && (right_inclusive == inclusive::NO))
        return detail::bin<T, thrust::less<T>, thrust::less<T>, false>(
          input, left_edges, right_edges, mr);
    }

    CUDF_FAIL("Undefined inclusive setting.");
  }
};

/// Bin the input by the edges in left_edges and right_edges.
std::unique_ptr<column> bin(column_view const& input,
                            column_view const& left_edges,
                            inclusive left_inclusive,
                            column_view const& right_edges,
                            inclusive right_inclusive,
                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE()
  CUDF_EXPECTS((input.type() == left_edges.type()) && (input.type() == right_edges.type()),
               "The input and edge columns must have the same types.");
  CUDF_EXPECTS(left_edges.size() == right_edges.size(),
               "The left and right edge columns must be of the same length.");
  // TODO: Decide whether to check `nullable` instead here. Allowing nullable
  // columns that don't actually contain any null values seems safe, but may
  // not be desirable from an API perspective.
  CUDF_EXPECTS(left_edges.null_count() == 0, "The left edges cannot contain nulls.");
  CUDF_EXPECTS(right_edges.null_count() == 0, "The right edges cannot contain nulls.");

  // Handle empty inputs.
  if (input.is_empty()) { return cudf::make_numeric_column(data_type(type_to_id<size_type>()), 0); }

  return type_dispatcher<dispatch_storage_type>(input.type(),
                                                bin_type_dispatcher{},
                                                input,
                                                left_edges,
                                                left_inclusive,
                                                right_edges,
                                                right_inclusive,
                                                mr);
}
}  // namespace cudf
