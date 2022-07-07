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

#include <vector>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

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

namespace {

/**
 * @brief Check if the input list has any null elements.
 *
 * @param list The input list.
 * @return The boolean result indicating if the input list has null elements.
 */
__device__ bool list_has_nulls(list_device_view list) {
  return thrust::any_of(thrust::seq, thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(list.size()),
                        [&list](auto const idx) { return list.is_null(idx); });
}

} // namespace

void post_process_list_overlap(cudf::column_view const &lhs, cudf::column_view const &rhs,
                               std::unique_ptr<cudf::column> const &overlap_result,
                               rmm::cuda_stream_view stream) {

  auto const overlap_cv = overlap_result->view();
  auto const lhs_cdv_ptr = column_device_view::create(lhs, stream);
  auto const rhs_cdv_ptr = column_device_view::create(rhs, stream);
  auto const overlap_cdv_ptr = column_device_view::create(overlap_cv, stream);

  // Create a new bitmask to satisfy Spark's arrays_overlap's special behavior.
  auto validity = rmm::device_uvector<bool>(overlap_cv.size(), stream);
  thrust::tabulate(rmm::exec_policy(stream), validity.begin(), validity.end(),
                   [lhs = cudf::detail::lists_column_device_view{*lhs_cdv_ptr},
                    rhs = cudf::detail::lists_column_device_view{*rhs_cdv_ptr},
                    overlap_result = *overlap_cdv_ptr] __device__(auto const idx) {
                     if (overlap_result.is_null(idx) ||
                         overlap_result.template element<bool>(idx)) {
                       return true;
                     }

                     // `lhs_list` and `rhs_list` should not be null, otherwise
                     // `overlap_result[idx]` is null and that has been handled above.
                     auto const lhs_list = list_device_view{lhs, idx};
                     auto const rhs_list = list_device_view{rhs, idx};

                     // Only proceed if both lists are non-empty.
                     if (lhs_list.size() == 0 || rhs_list.size() == 0) {
                       return true;
                     }

                     if (list_has_nulls(lhs_list) || list_has_nulls(rhs_list)) {
                       return false;
                     }
                   });

  auto const [null_mask, null_count] =
      cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity{});

  if (null_count > 0) {
    auto null_masks = std::vector<bitmask_type const *>{
        overlap_cv.null_mask(), static_cast<bitmask_type const *>(null_mask.data())};
    auto [new_null_mask, new_null_count] = cudf::detail::bitmask_and(
        null_masks, std::vector<cudf::size_type>{0, 0}, overlap_cv.size(), stream);
    overlap_result->set_null_mask(std::move(new_null_mask), new_null_count);
  }
}

} // namespace cudf::jni
