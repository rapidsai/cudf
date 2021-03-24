/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/logical.h>

namespace cudf {
namespace detail {
template <typename ColumnDeviceView, typename HostTableView>
void table_device_view_base<ColumnDeviceView, HostTableView>::destroy()
{
  delete _descendant_storage;
  delete this;
}

template <typename ColumnDeviceView, typename HostTableView>
table_device_view_base<ColumnDeviceView, HostTableView>::table_device_view_base(
  HostTableView source_view, rmm::cuda_stream_view stream)
  : _num_rows{source_view.num_rows()}, _num_columns{source_view.num_columns()}
{
  // The table's columns must be converted to ColumnDeviceView
  // objects and copied into device memory for the table_device_view's
  // _columns member.
  if (source_view.num_columns() > 0) {
    std::unique_ptr<rmm::device_buffer> descendant_storage_owner;
    std::tie(descendant_storage_owner, _columns) =
      contiguous_copy_column_device_views<ColumnDeviceView, HostTableView>(source_view, stream);
    _descendant_storage = descendant_storage_owner.release();
  }
}

// Explicit instantiation for a device table of immutable views
template class table_device_view_base<column_device_view, table_view>;

// Explicit instantiation for a device table of mutable views
template class table_device_view_base<mutable_column_device_view, mutable_table_view>;

namespace {
struct is_relationally_comparable_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return cudf::is_relationally_comparable<T, T>();
  }
};
}  // namespace

template <typename TableView>
bool is_relationally_comparable(TableView const& lhs, TableView const& rhs)
{
  return thrust::all_of(thrust::counting_iterator<size_type>(0),
                        thrust::counting_iterator<size_type>(lhs.num_columns()),
                        [lhs, rhs] __device__(auto const i) {
                          // Simplified this for compile time. (Ideally use double_type_dispatcher)
                          // TODO: possible to implement without double type dispatcher.
                          return lhs.column(i).type() == rhs.column(i).type() and
                                 type_dispatcher(lhs.column(i).type(),
                                                 is_relationally_comparable_impl{});
                        });
}

// Explicit extern template instantiation for a table of immutable views
extern template bool is_relationally_comparable<table_view>(table_view const& lhs,
                                                            table_view const& rhs);

// Explicit extern template instantiation for a table of mutable views
extern template bool is_relationally_comparable<mutable_table_view>(mutable_table_view const& lhs,
                                                                    mutable_table_view const& rhs);

// Explicit extern template instantiation for a device table of immutable views
template bool is_relationally_comparable<table_device_view>(table_device_view const& lhs,
                                                            table_device_view const& rhs);

// Explicit extern template instantiation for a device table of mutable views
template bool is_relationally_comparable<mutable_table_device_view>(
  mutable_table_device_view const& lhs, mutable_table_device_view const& rhs);

}  // namespace detail
}  // namespace cudf
