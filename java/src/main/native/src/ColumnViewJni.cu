/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/scan.h>

#include "ColumnViewJni.hpp"

namespace cudf::jni {

std::unique_ptr<cudf::column>
new_column_with_boolean_column_as_validity(cudf::column_view const &exemplar,
                                           cudf::column_view const &validity_column) {
  CUDF_EXPECTS(validity_column.type().id() == type_id::BOOL8,
               "Validity column must be of type bool");
  CUDF_EXPECTS(validity_column.size() == exemplar.size(),
               "Exemplar and validity columns must have the same size");

  auto validity_device_view = cudf::column_device_view::create(validity_column);
  auto validity_begin = cudf::detail::make_optional_iterator<bool>(
      *validity_device_view, cudf::nullate::DYNAMIC{validity_column.has_nulls()});
  auto validity_end = validity_begin + validity_device_view->size();
  auto [null_mask, null_count] =
      cudf::detail::valid_if(validity_begin, validity_end, [] __device__(auto optional_bool) {
        return optional_bool.value_or(false);
      });
  auto const exemplar_without_null_mask = cudf::column_view{
      exemplar.type(),
      exemplar.size(),
      exemplar.head<void>(),
      nullptr,
      0,
      exemplar.offset(),
      std::vector<cudf::column_view>{exemplar.child_begin(), exemplar.child_end()}};
  auto deep_copy = std::make_unique<cudf::column>(exemplar_without_null_mask);
  deep_copy->set_null_mask(std::move(null_mask), null_count);
  return deep_copy;
}

std::unique_ptr<cudf::column> generate_list_offsets(cudf::column_view const &list_length,
                                                    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(list_length.type().id() == cudf::type_id::INT32,
               "Input column does not have type INT32.");

  auto const begin_iter = list_length.template begin<cudf::size_type>();
  auto const end_iter = list_length.template end<cudf::size_type>();

  auto offsets_column = make_numeric_column(data_type{type_id::INT32}, list_length.size() + 1,
                                            mask_state::UNALLOCATED, stream);
  auto offsets_view = offsets_column->mutable_view();
  auto d_offsets = offsets_view.template begin<int32_t>();

  thrust::inclusive_scan(rmm::exec_policy(stream), begin_iter, end_iter, d_offsets + 1);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(int32_t), stream));

  return offsets_column;
}

std::unique_ptr<cudf::column> lists_distinct_by_key(cudf::lists_column_view const &input,
                                                    rmm::cuda_stream_view stream) {
  if (input.is_empty()) {
    return empty_like(input.parent());
  }

  auto const child = input.get_sliced_child(stream);

  // Genereate labels for the input list elements.
  auto labels = rmm::device_uvector<cudf::size_type>(child.size(), stream);
  cudf::detail::label_segments(input.offsets_begin(), input.offsets_end(), labels.begin(),
                               labels.end(), stream);

  // Use `cudf::duplicate_keep_option::KEEP_LAST` so this will produce the desired behavior when
  // being called in `create_map` in spark-rapids.
  // Other options comparing nulls and NaNs are set as all-equal.
  auto out_columns = cudf::detail::stable_distinct(
                         table_view{{column_view{cudf::device_span<cudf::size_type const>{labels}},
                                     child.child(0), child.child(1)}}, // input table
                         std::vector<size_type>{0, 1},                 // key columns
                         cudf::duplicate_keep_option::KEEP_LAST, cudf::null_equality::EQUAL,
                         cudf::nan_equality::ALL_EQUAL, stream)
                         ->release();
  auto const out_labels = out_columns.front()->view();

  // Assemble a structs column of <out_keys, out_vals>.
  auto out_structs_members = std::vector<std::unique_ptr<cudf::column>>();
  out_structs_members.emplace_back(std::move(out_columns[1]));
  out_structs_members.emplace_back(std::move(out_columns[2]));
  auto out_structs =
      cudf::make_structs_column(out_labels.size(), std::move(out_structs_members), 0, {});

  // Assemble a lists column of structs<out_keys, out_vals>.
  auto out_offsets = make_numeric_column(data_type{type_to_id<offset_type>()}, input.size() + 1,
                                         mask_state::UNALLOCATED, stream);
  auto const offsets_begin = out_offsets->mutable_view().template begin<offset_type>();
  auto const labels_begin = out_labels.template begin<offset_type>();
  cudf::detail::labels_to_offsets(labels_begin, labels_begin + out_labels.size(), offsets_begin,
                                  offsets_begin + out_offsets->size(), stream);

  return cudf::make_lists_column(input.size(), std::move(out_offsets), std::move(out_structs),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream), stream);
}

} // namespace cudf::jni
