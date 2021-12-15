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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>

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

} // namespace cudf::jni
