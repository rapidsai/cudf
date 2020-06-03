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

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/repeat.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>

#include <limits>
#include <memory>

namespace {
struct count_accessor {
  cudf::scalar const* p_scalar = nullptr;

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, cudf::size_type> operator()(cudaStream_t stream = 0)
  {
    using ScalarType = cudf::scalar_type_t<T>;
#if 1
    // TODO: temporary till cudf::scalar's value() function is marked as const
    auto p_count = const_cast<ScalarType*>(static_cast<ScalarType const*>(this->p_scalar));
#else
    auto p_count = static_cast<ScalarType const*>(this->p_scalar);
#endif
    auto count = p_count->value();
    // static_cast is necessary due to bool
    CUDF_EXPECTS(static_cast<int64_t>(count) <= std::numeric_limits<cudf::size_type>::max(),
                 "count should not exceed size_type's limit.");
    return static_cast<cudf::size_type>(count);
  }

  template <typename T>
  std::enable_if_t<not std::is_integral<T>::value, cudf::size_type> operator()(cudaStream_t stream)
  {
    CUDF_FAIL("count value should be a integral type.");
  }
};

struct compute_offsets {
  cudf::column_view const* p_column = nullptr;

  template <typename T>
  std::enable_if_t<std::is_integral<T>::value, rmm::device_vector<cudf::size_type>> operator()(
    bool check_count, cudaStream_t stream = 0)
  {
    // static_cast is necessary due to bool
    if (check_count && static_cast<int64_t>(std::numeric_limits<T>::max()) >
                         std::numeric_limits<cudf::size_type>::max()) {
      auto max = thrust::reduce(p_column->begin<T>(), p_column->end<T>(), 0, thrust::maximum<T>());
      CUDF_EXPECTS(max <= std::numeric_limits<cudf::size_type>::max(),
                   "count should not have values larger than size_type's limit.");
    }
    rmm::device_vector<cudf::size_type> offsets(p_column->size());
    thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
                           p_column->begin<T>(),
                           p_column->end<T>(),
                           offsets.begin());
    if (check_count == true) {
      CUDF_EXPECTS(thrust::is_sorted(
                     rmm::exec_policy(stream)->on(stream), offsets.begin(), offsets.end()) == true,
                   "count has negative values or the resulting table has more \
                    rows than size_type's limit.");
    }

    return offsets;
  }

  template <typename T>
  std::enable_if_t<not std::is_integral<T>::value, rmm::device_vector<cudf::size_type>> operator()(
    bool check_count, cudaStream_t stream)
  {
    CUDF_FAIL("count value should be a integral type.");
  }
};

}  // namespace

namespace cudf {
namespace detail {
std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              bool check_count,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream)
{
  CUDF_EXPECTS(input_table.num_rows() == count.size(), "in and count must have equal size");
  CUDF_EXPECTS(count.has_nulls() == false, "count cannot contain nulls");

  if (input_table.num_rows() == 0) { return cudf::empty_like(input_table); }

  auto offsets = cudf::type_dispatcher(count.type(), compute_offsets{&count}, check_count, stream);

  size_type output_size{offsets.back()};
  rmm::device_vector<size_type> indices(output_size);
  thrust::upper_bound(rmm::exec_policy(stream)->on(stream),
                      offsets.begin(),
                      offsets.end(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(output_size),
                      indices.begin());

  return gather(input_table, indices.begin(), indices.end(), false, mr, stream);
}

std::unique_ptr<table> repeat(table_view const& input_table,
                              size_type count,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream)
{
  CUDF_EXPECTS(count >= 0, "count value should be non-negative");
  CUDF_EXPECTS(
    static_cast<int64_t>(input_table.num_rows()) * count <= std::numeric_limits<size_type>::max(),
    "The resulting table has more rows than size_type's limit.");

  if ((input_table.num_rows() == 0) || (count == 0)) { return cudf::empty_like(input_table); }

  auto output_size = input_table.num_rows() * count;
  auto map_begin   = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), [count] __device__(auto i) { return i / count; });
  auto map_end = map_begin + output_size;

  return gather(input_table, map_begin, map_end, false, mr, stream);
}

}  // namespace detail

std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              bool check_count,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat(input_table, count, check_count, mr, 0);
}

std::unique_ptr<table> repeat(table_view const& input_table,
                              size_type count,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat(input_table, count, mr, 0);
}

}  // namespace cudf
