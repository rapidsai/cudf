/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/lists/drop_list_duplicates.hpp>
#include <cudf/lists/lists_column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {
namespace reduction {

std::unique_ptr<scalar> drop_duplicates(list_scalar const& scalar,
                                        null_equality nulls_equal,
                                        nan_equality nans_equal,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  auto list_wrapper   = lists::detail::make_lists_column_from_scalar(scalar, 1, stream, mr);
  auto lcw            = lists_column_view(list_wrapper->view());
  auto no_dup_wrapper = lists::drop_list_duplicates(lcw, nulls_equal, nans_equal, mr);
  auto no_dup         = lists_column_view(no_dup_wrapper->view()).get_sliced_child(stream);
  return make_list_scalar(no_dup, stream, mr);
}

std::unique_ptr<scalar> collect_list(column_view const& col,
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  if (null_handling == null_policy::EXCLUDE && col.has_nulls()) {
    auto d_view             = column_device_view::create(col, stream);
    auto filter             = detail::validity_accessor(*d_view);
    auto null_purged_table  = detail::copy_if(table_view{{col}}, filter, stream, mr);
    column* null_purged_col = null_purged_table->release().front().release();
    null_purged_col->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
    return std::make_unique<list_scalar>(std::move(*null_purged_col), true, stream, mr);
  } else {
    return make_list_scalar(col, stream, mr);
  }
}

std::unique_ptr<scalar> merge_lists(lists_column_view const& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto flatten_col = col.get_sliced_child(stream);
  return make_list_scalar(flatten_col, stream, mr);
}

std::unique_ptr<scalar> collect_set(column_view const& col,
                                    null_policy null_handling,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto scalar = collect_list(col, null_handling, stream, mr);
  auto ls     = dynamic_cast<list_scalar*>(scalar.get());
  return drop_duplicates(*ls, nulls_equal, nans_equal, stream, mr);
}

std::unique_ptr<scalar> merge_sets(lists_column_view const& col,
                                   null_equality nulls_equal,
                                   nan_equality nans_equal,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  auto flatten_col = col.get_sliced_child(stream);
  auto scalar      = std::make_unique<list_scalar>(flatten_col, true, stream, mr);
  return drop_duplicates(*scalar, nulls_equal, nans_equal, stream, mr);
}

}  // namespace reduction
}  // namespace cudf
