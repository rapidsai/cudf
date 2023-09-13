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

#include <reductions/hash_reduce_by_row.cuh>
#include <stream_compaction/stream_compaction_common.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <cuda/atomic>

namespace cudf::reduction::detail {

namespace {

/**
 * @brief The functor to compute the occurences of each unique rows in the input table.
 */
template <typename MapView, typename KeyHasher, typename KeyEqual, typename OutputType>
struct reduce_fn : cudf::detail::reduce_by_row_fn_base<MapView, KeyHasher, KeyEqual, OutputType> {
  reduce_fn(MapView const& d_map,
            KeyHasher const& d_hasher,
            KeyEqual const& d_equal,
            OutputType* const d_output)
    : cudf::detail::reduce_by_row_fn_base<MapView, KeyHasher, KeyEqual, OutputType>{
        d_map, d_hasher, d_equal, d_output}
  {
  }

  // Count the number of rows in each group of rows that are compared equal.
  __device__ void operator()(size_type const idx) const
  {
    cuda::atomic_ref<OutputType, cuda::thread_scope_device> count(*this->get_output_ptr(idx));
    count.fetch_add(OutputType{1}, cuda::std::memory_order_relaxed);
  }
};

/**
 * @brief The builder to construct an instance of `reduce_fn` functor.
 */
struct reduce_func_builder {
  template <typename MapView, typename KeyHasher, typename KeyEqual, typename OutputType>
  auto build(MapView const& d_map,
             KeyHasher const& d_hasher,
             KeyEqual const& d_equal,
             OutputType* const d_output)
  {
    return reduce_fn<MapView, KeyHasher, KeyEqual, OutputType>{d_map, d_hasher, d_equal, d_output};
  }
};

template <typename T>
struct is_none_zero {
  T const* data;
  __device__ bool operator()(size_type const idx) const { return data[idx] != T{0}; }
};

struct histogram_dispatcher {
  template <typename OutputType>
  static bool constexpr is_supported()
  {
    // Currently only int64_t is requested by Spark-Rapids.
    // More data type can be supported by enabling it below.
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
    mutable_column_view const& output,
    rmm::cuda_stream_view stream) const
  {
    auto const reduction_results =
      cudf::detail::hash_reduce_by_row(map,
                                       preprocessed_input,
                                       num_rows,
                                       has_nulls,
                                       has_nested_columns,
                                       null_equality::EQUAL,
                                       nan_equality::ALL_EQUAL,
                                       reduce_func_builder{},
                                       OutputType{0},
                                       stream,
                                       rmm::mr::get_current_device_resource());

    // Reduction results are either group sizes of equal rows, or `0`.
    // Thus, we only needs to extract the non-zero group sizes.
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(num_rows),
                    output.begin<OutputType>(),
                    is_none_zero<OutputType>{reduction_results.begin()});
  }
};

}  // namespace

std::unique_ptr<cudf::column> histogram(table_view const& input,
                                        data_type const output_dtype,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(cudf::is_integral(output_dtype) &&
                 (cudf::size_of(output_dtype) == 4 || cudf::size_of(output_dtype) == 8),
               "The output type of histogram aggregation must be an 32/64bit integral type.");

  auto map = cudf::detail::hash_map_type{
    compute_hash_table_size(input.num_rows()),
    cuco::empty_key{cudf::detail::COMPACTION_EMPTY_KEY_SENTINEL},
    cuco::empty_value{cudf::detail::COMPACTION_EMPTY_VALUE_SENTINEL},
    cudf::detail::hash_table_allocator_type{default_allocator<char>{}, stream},
    stream.value()};

  auto const preprocessed_input =
    cudf::experimental::row::hash::preprocessed_table::create(input, stream);
  auto const has_nulls          = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const has_nested_columns = cudf::detail::has_nested_columns(input);

  auto const row_hasher = cudf::experimental::row::hash::row_hasher(preprocessed_input);
  auto const key_hasher =
    cudf::detail::experimental::compaction_hash(row_hasher.device_hasher(has_nulls));
  auto const row_comp = cudf::experimental::row::equality::self_comparator(preprocessed_input);

  auto const pair_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, [] __device__(size_type const i) { return cuco::make_pair(i, i); });

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
  auto distinct_indices = cudf::make_numeric_column(data_type{type_to_id<size_type>()},
                                                    static_cast<size_type>(map.get_size()),
                                                    mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);
  map.retrieve_all(distinct_indices->mutable_view().begin<size_type>(),
                   thrust::make_discard_iterator(),
                   stream.value());

  // Count the number of occurences of each unique row.
  auto unique_counts = make_numeric_column(
    output_dtype, static_cast<size_type>(map.get_size()), mask_state::UNALLOCATED, stream, mr);
  type_dispatcher(output_dtype,
                  histogram_dispatcher{},
                  map,
                  std::move(preprocessed_input),
                  input.num_rows(),
                  has_nulls,
                  has_nested_columns,
                  unique_counts->mutable_view(),
                  stream);

  std::vector<std::unique_ptr<column>> output_children;
  output_children.emplace_back(std::move(distinct_indices));
  output_children.emplace_back(std::move(unique_counts));

  return make_structs_column(
    static_cast<size_type>(map.get_size()), std::move(output_children), 0, {}, stream, mr);
}

std::unique_ptr<cudf::column> histogram(column_view const& input,
                                        data_type const output_dtype,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  return histogram(table_view{{input}}, output_dtype, stream, mr);
}

std::unique_ptr<cudf::column> merge_histogram(column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(
    input.type().id() == type_id::STRUCT && input.num_children() == 2,
    "The input of merge_histogram aggregation must be a struct column having two children.");
  CUDF_EXPECTS(cudf::is_integral(input.child(1).type()),
               "The second child of the input column must be an integer type.");

  return nullptr;
}

}  // namespace cudf::reduction::detail
