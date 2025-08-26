/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "create_results_table.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby::detail::hash {

std::pair<key_map_t, rmm::device_uvector<size_type>> find_output_indices(
  device_span<size_type> key_indices,
  device_span<size_type const> unique_indices,
  rmm::cuda_stream_view stream)
{
  cudf::scoped_range r("find output_indices");

  // {
  //   auto tmp = cudf::detail::make_std_vector(unique_indices, stream);
  //   std::sort(tmp.begin(), tmp.end());
  //   printf("unique keys: %d\n", (int)tmp.size());
  //   for (auto i : tmp) {
  //     printf("%d, ", (int)i);
  //   }
  //   printf("\n");
  //   fflush(stdout);
  // }

  auto tmp = rmm::device_uvector<size_type>(key_indices.size(), stream);
  thrust::copy(
    rmm::exec_policy_nosync(stream), key_indices.begin(), key_indices.end(), tmp.begin());
  stream.synchronize();

  // auto set = cuco::static_set{
  //   cuco::extent<int64_t>{unique_indices.size()},
  //   cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% load factor
  //   cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
  //   key_indices_comparator_t{unique_indices.begin()},
  //   simplified_probing_scheme_t{unique_indices.begin()},
  //   cuco::thread_scope_device,
  //   cuco::storage<GROUPBY_BUCKET_SIZE>{},
  //   cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
  //   stream.value()};

  rmm::device_uvector<size_type> counts(2, stream);
  thrust::fill(rmm::exec_policy_nosync(stream), counts.begin(), counts.end(), 0);
  stream.synchronize();

  auto map =
    key_map_t{cuco::extent<std::size_t>{unique_indices.size()},
              0.5,
              cuco::empty_key<cudf::size_type>{-1},
              cuco::empty_value<cudf::size_type>{-1},
              // eq_t{counts.begin()},
              // cuco::linear_probing<GROUPBY_CG_SIZE, hash_t>{hash_t{counts.begin() + 1}},
              {},
              {},
              cuco::thread_scope_device,
              cuco::storage<GROUPBY_BUCKET_SIZE>{},
              cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
              stream.value()};

  rmm::device_uvector<cudf::size_type> new_indices(key_indices.size(), stream);
  if (unique_indices.size() == 0) {
    // return new_indices;
    return std::pair{std::move(map), std::move(new_indices)};
  }

  stream.synchronize();

  // #if 0
  {
    cudf::scoped_range r("scatter");
    thrust::scatter(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(static_cast<size_type>(unique_indices.size())),
                    unique_indices.begin(),
                    new_indices.begin());
    stream.synchronize();
  }
  {
    cudf::scoped_range r("update indices");
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(0),
                       static_cast<size_type>(key_indices.size()),
                       [new_indices = new_indices.begin(),
                        key_indices = key_indices.begin()] __device__(size_type const idx) {
                         if (key_indices[idx] != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
                           key_indices[idx] = new_indices[key_indices[idx]];
                         }
                       });
    stream.synchronize();
  }

  {
    cudf::scoped_range r("clear");
    thrust::copy(rmm::exec_policy_nosync(stream), tmp.begin(), tmp.end(), key_indices.begin());
    stream.synchronize();
  }

  // return new_indices;
  // #else
  {
    cudf::scoped_range r("insert");
    auto it = cudf::detail::make_counting_transform_iterator(
      0, [unique_indices = unique_indices.begin()] __device__(size_type const idx) {
        return cuco::make_pair(unique_indices[idx], idx);
      });
    map.insert_async(it, it + unique_indices.size(), stream.value());
    // auto c = map.insert(it, it + unique_indices.size(), stream.value());
    // printf("inserted: %d\n", (int)c);
    // fflush(stdout);
    stream.synchronize();
  }

  {
    cudf::scoped_range r("update indices");
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(0),
                       static_cast<size_type>(key_indices.size()),
                       [map         = map.ref(cuco::op::find),
                        key_indices = key_indices.begin()] __device__(size_type const idx) {
                         if (key_indices[idx] != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
                           auto const itr   = map.find(key_indices[idx]);
                           key_indices[idx] = (itr != map.end())
                                                ? itr->second
                                                : cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
                         }
                       });
    stream.synchronize();
  }

  return std::pair{std::move(map), std::move(new_indices)};
  // return map;
  // #endif
}

namespace {
/**
 * @brief Functor to create sparse result columns for hash-based groupby aggregations
 *
 * This functor handles the creation of appropriately typed and sized columns for each
 * aggregation, including special handling for SUM_WITH_OVERFLOW which requires a struct column.
 */
struct sparse_column_creator {
  cudf::size_type output_size;
  rmm::cuda_stream_view stream;

  explicit sparse_column_creator(cudf::size_type output_size, rmm::cuda_stream_view stream)
    : output_size{output_size}, stream{stream}
  {
  }

  std::unique_ptr<cudf::column> operator()(cudf::column_view const& col,
                                           cudf::aggregation::Kind const& agg) const
  {
    auto const nullable =
      (agg == cudf::aggregation::COUNT_VALID or agg == cudf::aggregation::COUNT_ALL)
        ? false
        : (col.has_nulls() or agg == cudf::aggregation::VARIANCE or agg == cudf::aggregation::STD);
    auto const mask_flag = (nullable) ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED;
    auto const col_type  = cudf::is_dictionary(col.type())
                             ? cudf::dictionary_column_view(col).keys().type()
                             : col.type();

    // Special handling for SUM_WITH_OVERFLOW which needs a struct column
    if (agg == cudf::aggregation::SUM_WITH_OVERFLOW) {
      // Lambda to create empty columns for better readability
      auto make_empty_column = [&stream = this->stream](cudf::type_id type_id,
                                                        cudf::size_type size,
                                                        cudf::mask_state mask_state) {
        return make_fixed_width_column(cudf::data_type{type_id}, size, mask_state, stream);
      };

      // TODO: fix mask state
      // Lambda to create children for SUM_WITH_OVERFLOW struct column
      auto make_children = [&make_empty_column](cudf::size_type size, cudf::mask_state mask_state) {
        std::vector<std::unique_ptr<cudf::column>> children;
        // Create sum child column (int64_t) - no null mask needed, struct-level mask handles
        // nullability
        children.push_back(
          make_empty_column(cudf::type_id::INT64, size, cudf::mask_state::UNALLOCATED));
        // Create overflow child column (bool) - no null mask needed, only value matters
        children.push_back(
          make_empty_column(cudf::type_id::BOOL8, size, cudf::mask_state::UNALLOCATED));
        return children;
      };

      if (output_size == 0) {
        // For empty columns, create empty struct column manually
        auto children = make_children(0, cudf::mask_state::UNALLOCATED);
        return create_structs_hierarchy(0, std::move(children), 0, {}, stream);
      } else {
        auto children = make_children(output_size, mask_flag);

        // Create struct column with the children
        // For SUM_WITH_OVERFLOW, make struct nullable if input has nulls (same as other
        // aggregations)
        if (nullable) {
          // Start with ALL_NULL, results will be marked valid during aggregation
          auto null_mask  = cudf::create_null_mask(output_size, cudf::mask_state::ALL_NULL, stream);
          auto null_count = output_size;  // All null initially
          return create_structs_hierarchy(
            output_size, std::move(children), null_count, std::move(null_mask), stream);
        } else {
          return create_structs_hierarchy(output_size, std::move(children), 0, {}, stream);
        }
      }
    } else {
      return make_fixed_width_column(
        cudf::detail::target_type(col_type, agg), output_size, mask_flag, stream);
    }
  }
};
}  // anonymous namespace

template <typename SetType>
void extract_populated_keys(SetType const& key_set,
                            rmm::device_uvector<cudf::size_type>& populated_keys,
                            rmm::cuda_stream_view stream)
{
  cudf::scoped_range r("extract keys");
  auto const keys_end = key_set.retrieve_all(populated_keys.begin(), stream.value());

  populated_keys.resize(std::distance(populated_keys.begin(), keys_end), stream);
  stream.synchronize();
}

cudf::table create_results_table(cudf::size_type output_size,
                                 cudf::table_view const& flattened_values,
                                 host_span<cudf::aggregation::Kind const> agg_kinds,
                                 rmm::cuda_stream_view stream)
{
  cudf::scoped_range r("create results table");
  std::vector<std::unique_ptr<cudf::column>> output_cols;
  std::transform(flattened_values.begin(),
                 flattened_values.end(),
                 agg_kinds.begin(),
                 std::back_inserter(output_cols),
                 sparse_column_creator{output_size, stream});
  cudf::table result_table(std::move(output_cols));
  cudf::mutable_table_view result_table_view = result_table.mutable_view();
  cudf::detail::initialize_with_identity(result_table_view, agg_kinds, stream);
  stream.synchronize();
  return result_table;
}

template void extract_populated_keys<global_set_t>(
  global_set_t const& key_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);

template void extract_populated_keys<nullable_global_set_t>(
  nullable_global_set_t const& key_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);

}  // namespace cudf::groupby::detail::hash
