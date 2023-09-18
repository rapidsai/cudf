/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <stream_compaction/stream_compaction_common.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/hash_reduce_by_row.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <cuda/atomic>

#include <optional>

namespace cudf::reduction::detail {

namespace {

/**
 * @brief The functor to accumulate the frequency of each distinct rows in the input table.
 */
template <typename MapView, typename KeyHasher, typename KeyEqual, typename OutputType>
struct reduce_fn : cudf::detail::reduce_by_row_fn_base<MapView, KeyHasher, KeyEqual, OutputType> {
  OutputType const* d_partial_output;

  reduce_fn(MapView const& d_map,
            KeyHasher const& d_hasher,
            KeyEqual const& d_equal,
            OutputType* const d_output,
            OutputType const* const d_partial_output)
    : cudf::detail::reduce_by_row_fn_base<MapView, KeyHasher, KeyEqual, OutputType>{d_map,
                                                                                    d_hasher,
                                                                                    d_equal,
                                                                                    d_output},
      d_partial_output{d_partial_output}
  {
  }

  // Count the number of rows in each group of rows that are compared equal.
  __device__ void operator()(size_type const idx) const
  {
    auto const increment = d_partial_output ? d_partial_output[idx] : OutputType{1};
    auto const count =
      cuda::atomic_ref<OutputType, cuda::thread_scope_device>(*this->get_output_ptr(idx));
    count.fetch_add(increment, cuda::std::memory_order_relaxed);
  }
};

/**
 * @brief The builder to construct an instance of `reduce_fn` functor.
 */
template <typename OutputType>
struct reduce_func_builder {
  OutputType const* const d_partial_output;

  reduce_func_builder(OutputType const* const d_partial_output) : d_partial_output{d_partial_output}
  {
  }

  template <typename MapView, typename KeyHasher, typename KeyEqual>
  auto build(MapView const& d_map,
             KeyHasher const& d_hasher,
             KeyEqual const& d_equal,
             OutputType* const d_output)
  {
    return reduce_fn<MapView, KeyHasher, KeyEqual, OutputType>{
      d_map, d_hasher, d_equal, d_output, d_partial_output};
  }
};

/**
 * @brief Specialized functor to check for non-zero.
 *
 * The input must be given as Pair<T1, T2>. Only value of T2 is checked for non-zero.
 */
struct is_none_zero {
  template <typename Pair>
  __device__ bool operator()(Pair const input) const
  {
    return thrust::get<1>(input) != 0;
  }
};

/**
 * @brief Dispatcher functor to compute histogram in the given OutputType.
 *
 * The indices of distinct rows and their corresponding frequencies are written into two separate
 * output buffer.
 */
struct histogram_dispatcher {
  template <typename OutputType>
  static bool constexpr is_supported()
  {
    // Currently only int64_t is requested by Spark-Rapids.
    // More data type (integer only) can be supported by enabling below.
    return std::is_same_v<OutputType, int64_t>;
  }

  template <typename OutputType, typename... Args>
  std::enable_if_t<!is_supported<OutputType>(), void> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported output type in histogram aggregation.");
  }

  template <typename OutputType, CUDF_ENABLE_IF(is_supported<OutputType>())>
  void operator()(
    cudf::detail::hash_map_type const& map,
    std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const preprocessed_input,
    size_type num_rows,
    cudf::nullate::DYNAMIC has_nulls,
    bool has_nested_columns,
    size_type* output_indices,
    mutable_column_view const& output_counts,
    std::optional<column_view> const& partial_counts,
    rmm::cuda_stream_view stream) const
  {
    auto const reduction_results = cudf::detail::hash_reduce_by_row(
      map,
      preprocessed_input,
      num_rows,
      has_nulls,
      has_nested_columns,
      null_equality::EQUAL,
      nan_equality::ALL_EQUAL,
      reduce_func_builder<OutputType>{partial_counts ? partial_counts.value().begin<OutputType>()
                                                     : nullptr},
      OutputType{0},
      stream,
      rmm::mr::get_current_device_resource());

    auto const input_it = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::make_counting_iterator(0), reduction_results.begin()));
    auto const output_it = thrust::make_zip_iterator(
      thrust::make_tuple(output_indices, output_counts.begin<OutputType>()));

    // Reduction results above are either group sizes of equal rows, or `0`.
    // Thus, we need to extract the non-zero group sizes.
    thrust::copy_if(
      rmm::exec_policy(stream), input_it, input_it + num_rows, output_it, is_none_zero{});
  }
};

auto gather_histogram(table_view const& input,
                      device_span<size_type const> distinct_indices,
                      std::unique_ptr<column>&& distinct_counts,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
{
  auto distinct_rows =
    std::move(cudf::detail::gather(input,
                                   distinct_indices,
                                   out_of_bounds_policy::DONT_CHECK,
                                   cudf::detail::negative_index_policy::NOT_ALLOWED,
                                   stream,
                                   mr)
                ->release()
                .front());

  std::vector<std::unique_ptr<column>> struct_children;
  struct_children.emplace_back(std::move(distinct_rows));
  struct_children.emplace_back(std::move(distinct_counts));
  auto output_structs = make_structs_column(
    static_cast<size_type>(distinct_indices.size()), std::move(struct_children), 0, {}, stream, mr);

  return std::make_unique<cudf::list_scalar>(
    std::move(*output_structs.release()), true, stream, mr);
}

}  // namespace

std::pair<rmm::device_uvector<size_type>, std::unique_ptr<column>> table_histogram(
  table_view const& input,
  std::optional<column_view> const& partial_counts,
  data_type const output_dtype,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_integral(output_dtype),
               "The output type of histogram aggregation must be an integral type.");

  auto map = cudf::detail::hash_map_type{
    compute_hash_table_size(input.num_rows()),
    cuco::empty_key{-1},
    cuco::empty_value{std::numeric_limits<size_type>::min()},
    cudf::detail::hash_table_allocator_type{default_allocator<char>{}, stream},
    stream.value()};

  auto const preprocessed_input =
    cudf::experimental::row::hash::preprocessed_table::create(input, stream);
  auto const has_nulls          = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const has_nested_columns = cudf::detail::has_nested_columns(input);

  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const key_hasher = row_hasher.device_hasher(has_nulls);
  auto const row_comp   = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  auto const pair_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return cuco::make_pair(i, i); });

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
  auto distinct_indices = rmm::device_uvector<size_type>(
    static_cast<size_type>(map.get_size()), stream, rmm::mr::get_current_device_resource());

  // Store the number of occurences of each distinct row.
  auto distinct_counts = make_numeric_column(
    output_dtype, static_cast<size_type>(map.get_size()), mask_state::UNALLOCATED, stream, mr);

  type_dispatcher(output_dtype,
                  histogram_dispatcher{},
                  map,
                  std::move(preprocessed_input),
                  input.num_rows(),
                  has_nulls,
                  has_nested_columns,
                  distinct_indices.begin(),
                  distinct_counts->mutable_view(),
                  partial_counts,
                  stream);

  return {std::move(distinct_indices), std::move(distinct_counts)};
}

std::unique_ptr<cudf::scalar> histogram(column_view const& input,
                                        data_type const output_dtype,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  auto const input_tv = table_view{{input}};
  auto [distinct_indices, distinct_counts] =
    table_histogram(input_tv, std::nullopt, output_dtype, stream, mr);
  return gather_histogram(input_tv, distinct_indices, std::move(distinct_counts), stream, mr);
}

std::unique_ptr<cudf::scalar> merge_histogram(column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(!input.has_nulls(), "The input column must not have nulls.");
  CUDF_EXPECTS(
    input.type().id() == type_id::STRUCT && input.num_children() == 2,
    "The input of merge_histogram aggregation must be a struct column having two children.");
  CUDF_EXPECTS(cudf::is_integral(input.child(1).type()) && !input.child(1).has_nulls(),
               "The second child of the input column must be ingegral type and has no nulls.");

  auto const structs_cv   = structs_column_view{input};
  auto const input_values = structs_cv.get_sliced_child(0, stream);
  auto const input_counts = structs_cv.get_sliced_child(1, stream);

  auto const values_tv = table_view{{input_values}};
  auto [distinct_indices, distinct_counts] =
    table_histogram(values_tv, input_counts, data_type{type_id::INT64}, stream, mr);
  return gather_histogram(values_tv, distinct_indices, std::move(distinct_counts), stream, mr);
}

}  // namespace cudf::reduction::detail
