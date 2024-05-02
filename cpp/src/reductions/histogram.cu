/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/gather.hpp>
#include <cudf/detail/hash_reduce_by_row.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <optional>

namespace cudf::reduction::detail {

namespace {

// Always use 64-bit signed integer for storing count.
using histogram_count_type = int64_t;

/**
 * @brief The functor to accumulate the frequency of each distinct rows in the input table.
 */
template <typename MapView, typename KeyHasher, typename KeyEqual, typename CountType>
struct reduce_fn : cudf::detail::reduce_by_row_fn_base<MapView, KeyHasher, KeyEqual, CountType> {
  CountType const* d_partial_output;

  reduce_fn(MapView const& d_map,
            KeyHasher const& d_hasher,
            KeyEqual const& d_equal,
            CountType* const d_output,
            CountType const* const d_partial_output)
    : cudf::detail::reduce_by_row_fn_base<MapView, KeyHasher, KeyEqual, CountType>{d_map,
                                                                                   d_hasher,
                                                                                   d_equal,
                                                                                   d_output},
      d_partial_output{d_partial_output}
  {
  }

  // Count the number of rows in each group of rows that are compared equal.
  __device__ void operator()(size_type const idx) const
  {
    auto const increment = d_partial_output ? d_partial_output[idx] : CountType{1};
    auto const count =
      cuda::atomic_ref<CountType, cuda::thread_scope_device>(*this->get_output_ptr(idx));
    count.fetch_add(increment, cuda::std::memory_order_relaxed);
  }
};

/**
 * @brief The builder to construct an instance of `reduce_fn` functor.
 */
template <typename CountType>
struct reduce_func_builder {
  CountType const* const d_partial_output;

  reduce_func_builder(CountType const* const d_partial_output) : d_partial_output{d_partial_output}
  {
  }

  template <typename MapView, typename KeyHasher, typename KeyEqual>
  auto build(MapView const& d_map,
             KeyHasher const& d_hasher,
             KeyEqual const& d_equal,
             CountType* const d_output)
  {
    return reduce_fn<MapView, KeyHasher, KeyEqual, CountType>{
      d_map, d_hasher, d_equal, d_output, d_partial_output};
  }
};

/**
 * @brief Specialized functor to check for not-zero of the second component of the input.
 */
struct is_not_zero {
  template <typename Pair>
  __device__ bool operator()(Pair const input) const
  {
    return thrust::get<1>(input) != 0;
  }
};

/**
 * @brief Building a histogram by gathering distinct rows from the input table and their
 * corresponding distinct counts.
 *
 * @param input The input table
 * @param distinct_indices Indices of the distinct rows
 * @param distinct_counts Distinct counts corresponding to the distinct rows
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object's device memory
 * @return A list_scalar storing the output histogram
 */
auto gather_histogram(table_view const& input,
                      device_span<size_type const> distinct_indices,
                      std::unique_ptr<column>&& distinct_counts,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  auto distinct_rows = cudf::detail::gather(input,
                                            distinct_indices,
                                            out_of_bounds_policy::DONT_CHECK,
                                            cudf::detail::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            mr);

  std::vector<std::unique_ptr<column>> struct_children;
  struct_children.emplace_back(std::move(distinct_rows->release().front()));
  struct_children.emplace_back(std::move(distinct_counts));
  auto output_structs = make_structs_column(
    static_cast<size_type>(distinct_indices.size()), std::move(struct_children), 0, {}, stream, mr);

  return std::make_unique<cudf::list_scalar>(
    std::move(*output_structs.release()), true, stream, mr);
}

}  // namespace

std::unique_ptr<column> make_empty_histogram_like(column_view const& values)
{
  std::vector<std::unique_ptr<column>> struct_children;
  struct_children.emplace_back(empty_like(values));
  struct_children.emplace_back(make_numeric_column(data_type{type_id::INT64}, 0));
  return std::make_unique<column>(data_type{type_id::STRUCT},
                                  0,
                                  rmm::device_buffer{},
                                  rmm::device_buffer{},
                                  0,
                                  std::move(struct_children));
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>, std::unique_ptr<column>>
compute_row_frequencies(table_view const& input,
                        std::optional<column_view> const& partial_counts,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  auto const has_nested_columns = cudf::detail::has_nested_columns(input);

  // Nested types are not tested, thus we just throw exception if we see such input for now.
  // We should remove this check after having enough tests.
  CUDF_EXPECTS(!has_nested_columns,
               "Nested types are not yet supported in histogram aggregation.",
               std::invalid_argument);

  auto map = cudf::detail::hash_map_type{compute_hash_table_size(input.num_rows()),
                                         cuco::empty_key{-1},
                                         cuco::empty_value{std::numeric_limits<size_type>::min()},
                                         cudf::detail::cuco_allocator{stream},
                                         stream.value()};

  auto const preprocessed_input =
    cudf::experimental::row::hash::preprocessed_table::create(input, stream);
  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(input)};

  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const key_hasher = row_hasher.device_hasher(has_nulls);
  auto const row_comp   = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  auto const pair_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0},
    cuda::proclaim_return_type<cuco::pair<size_type, size_type>>(
      [] __device__(size_type const i) { return cuco::make_pair(i, i); }));

  // Always compare NaNs as equal.
  using nan_equal_comparator =
    cudf::experimental::row::equality::nan_equal_physical_equality_comparator;
  auto const value_comp = nan_equal_comparator{};

  if (has_nested_columns) {
    auto const key_equal = row_comp.equal_to<true>(has_nulls, null_equality::EQUAL, value_comp);
    map.insert(pair_iter, pair_iter + input.num_rows(), key_hasher, key_equal, stream.value());
  } else {
    auto const key_equal = row_comp.equal_to<false>(has_nulls, null_equality::EQUAL, value_comp);
    map.insert(pair_iter, pair_iter + input.num_rows(), key_hasher, key_equal, stream.value());
  }

  // Gather the indices of distinct rows.
  auto distinct_indices = std::make_unique<rmm::device_uvector<size_type>>(
    static_cast<size_type>(map.get_size()), stream, mr);

  // Store the number of occurrences of each distinct row.
  auto distinct_counts = make_numeric_column(data_type{type_to_id<histogram_count_type>()},
                                             static_cast<size_type>(map.get_size()),
                                             mask_state::UNALLOCATED,
                                             stream,
                                             mr);

  // Compute frequencies (aka distinct counts) for the input rows.
  // Note that we consider null and NaNs as always equal.
  auto const reduction_results = cudf::detail::hash_reduce_by_row(
    map,
    preprocessed_input,
    input.num_rows(),
    has_nulls,
    has_nested_columns,
    null_equality::EQUAL,
    nan_equality::ALL_EQUAL,
    reduce_func_builder<histogram_count_type>{
      partial_counts ? partial_counts.value().begin<histogram_count_type>() : nullptr},
    histogram_count_type{0},
    stream,
    rmm::mr::get_current_device_resource());

  auto const input_it = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_counting_iterator(0), reduction_results.begin()));
  auto const output_it = thrust::make_zip_iterator(thrust::make_tuple(
    distinct_indices->begin(), distinct_counts->mutable_view().begin<histogram_count_type>()));

  // Reduction results above are either group sizes of equal rows, or `0`.
  // The final output is non-zero group sizes only.
  thrust::copy_if(
    rmm::exec_policy(stream), input_it, input_it + input.num_rows(), output_it, is_not_zero{});

  return {std::move(distinct_indices), std::move(distinct_counts)};
}

std::unique_ptr<cudf::scalar> histogram(column_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  // Empty group should be handled before reaching here.
  CUDF_EXPECTS(input.size() > 0, "Input should not be empty.", std::invalid_argument);

  auto const input_tv = table_view{{input}};
  auto [distinct_indices, distinct_counts] =
    compute_row_frequencies(input_tv, std::nullopt, stream, mr);
  return gather_histogram(input_tv, *distinct_indices, std::move(distinct_counts), stream, mr);
}

std::unique_ptr<cudf::scalar> merge_histogram(column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  // Empty group should be handled before reaching here.
  CUDF_EXPECTS(input.size() > 0, "Input should not be empty.", std::invalid_argument);
  CUDF_EXPECTS(!input.has_nulls(), "The input column must not have nulls.", std::invalid_argument);
  CUDF_EXPECTS(input.type().id() == type_id::STRUCT && input.num_children() == 2,
               "The input must be a structs column having two children.",
               std::invalid_argument);
  CUDF_EXPECTS(cudf::is_integral(input.child(1).type()) && !input.child(1).has_nulls(),
               "The second child of the input column must be of integral type and without nulls.",
               std::invalid_argument);

  auto const structs_cv   = structs_column_view{input};
  auto const input_values = structs_cv.get_sliced_child(0, stream);
  auto const input_counts = structs_cv.get_sliced_child(1, stream);

  auto const values_tv = table_view{{input_values}};
  auto [distinct_indices, distinct_counts] =
    compute_row_frequencies(values_tv, input_counts, stream, mr);
  return gather_histogram(values_tv, *distinct_indices, std::move(distinct_counts), stream, mr);
}

}  // namespace cudf::reduction::detail
