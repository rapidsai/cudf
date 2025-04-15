/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/detail/dremel.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/gather.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <optional>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

class sort_merge_join {
 public:
  sort_merge_join(table_view const &left, bool is_left_sorted, table_view const &right, bool is_right_sorted,
                    null_equality compare_nulls,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
    inner_join(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  struct preprocessed_table {
    table_view raw_tbl_view;
    table_view tbl_view;
    // filters for null_equality::UNEQUAL
    std::optional<rmm::device_buffer> raw_validity_mask = std::nullopt;
    std::optional<size_type> raw_num_nulls = std::nullopt;
    std::optional<std::unique_ptr<table>> tbl = std::nullopt;
    // optional reordering if we are given pre-sorted tables
    std::optional<std::unique_ptr<column>> tbl_sorted_order = std::nullopt;

    void populate_nonnull_filter(rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);
    void apply_nonnull_filter(rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);
    void preprocess_raw_table(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
    void get_sorted_order(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
    rmm::device_uvector<size_type> map_tbl_to_raw(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
  };
 private:
  preprocessed_table ptleft;
  preprocessed_table ptright;
  null_equality compare_nulls;

  void preprocess_tables(table_view const left,
                  table_view const right,
                  null_equality compare_nulls,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr);
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  postprocess_indices(
                      std::unique_ptr<rmm::device_uvector<size_type>> smaller_indices,
                      std::unique_ptr<rmm::device_uvector<size_type>> larger_indices,
                      null_equality compare_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);
};

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
