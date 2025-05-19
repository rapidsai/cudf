/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cassert>
#include <memory>
#include <numeric>

/**
 * @file table_device_view.cuh
 * @brief Table device view class definitions
 */

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Base class for a device table of `ColumnDeviceView`s
 *
 * This class should not be used directly. Instead:
 * - `table_device_view` should be used for a table of columns of type `column_device_view`
 * - `mutable_table_device_view` should be used for a table of columns of type
 * `mutable_column_device_view`
 *
 * All public constructors and member functions of `table_device_view_base` are
 * available in both `table_device_view` and `mutable_table_device_view`.
 *
 * @tparam ColumnDeviceView The type of column device view the table contains:
 *                          expects column_device_view or mutable_column_device_view
 * @tparam HostTableView The type of table view used to create the table device view:
 *                       expects table_view or mutable_table_view
 */
template <typename ColumnDeviceView, typename HostTableView>
class table_device_view_base {
 public:
  table_device_view_base()                              = delete;
  ~table_device_view_base()                             = default;
  table_device_view_base(table_device_view_base const&) = default;  ///< Copy constructor
  table_device_view_base(table_device_view_base&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  table_device_view_base& operator=(table_device_view_base const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  table_device_view_base& operator=(table_device_view_base&&) = default;

  /**
   * @brief Returns an iterator to the first view in the `table`.
   *
   * @return An iterator to the first view in the `table`
   */
  __device__ ColumnDeviceView* begin() const noexcept { return _columns; }

  /**
   * @brief Returns an iterator one past the last column view in the `table`.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *
   * @return An iterator to one past the last column view in the `table`
   */
  __device__ ColumnDeviceView* end() const noexcept { return _columns + _num_columns; }

  /**
   * @brief Returns a reference to the view of the specified column
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   */
  __device__ ColumnDeviceView const& column(size_type column_index) const noexcept
  {
    assert(column_index >= 0);
    assert(column_index < _num_columns);
    return _columns[column_index];
  }

  /**
   * @brief Returns a reference to the view of the specified column
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   */
  __device__ ColumnDeviceView& column(size_type column_index) noexcept
  {
    assert(column_index >= 0);
    assert(column_index < _num_columns);
    return _columns[column_index];
  }

  /**
   * @brief Returns the number of columns
   *
   * @return The number of columns
   */
  [[nodiscard]] __host__ __device__ size_type num_columns() const noexcept { return _num_columns; }

  /**
   * @brief Returns the number of rows
   *
   * @return The number of rows
   */
  [[nodiscard]] __host__ __device__ size_type num_rows() const noexcept { return _num_rows; }

  /**
   * @brief Destroy the `table_device_view` object.
   *
   * @note Does not free the table data, simply frees the device memory
   * allocated to hold the constituent column views.
   */
  void destroy();

 private:
  ColumnDeviceView* _columns{};  ///< Array of view objects in device memory
  size_type _num_rows{};
  size_type _num_columns{};

 protected:
  /**
   * @brief Construct a new table device view base object from host table_view
   *
   * @param source_view The host table_view to create table device view from
   * @param stream The CUDA stream to use for device memory allocation
   */
  table_device_view_base(HostTableView source_view, rmm::cuda_stream_view stream);

  /// Pointer to device memory holding the descendant storage
  rmm::device_buffer* _descendant_storage{};
};
}  // namespace detail

/**
 * @brief Table device view that is usable in device memory
 */
class table_device_view : public detail::table_device_view_base<column_device_view, table_view> {
 public:
  /**
   * @brief Factory to construct a table device view that is usable in device memory.
   *
   * Allocates and copies views of `source_view`'s children to device memory to
   * make them accessible in device code.
   *
   * Returns a `std::unique_ptr<table_device_view>` with a custom deleter to
   * free the device memory allocated for the children.
   *
   * @param source_view The table view whose contents will be copied to create a new table
   * @param stream CUDA stream used for device memory operations
   * @return A `unique_ptr` to a `table_device_view` that makes the data from `source_view`
   * available in device memory
   */
  static auto create(table_view source_view,
                     rmm::cuda_stream_view stream = cudf::get_default_stream())
  {
    auto deleter = [](table_device_view* t) { t->destroy(); };
    return std::unique_ptr<table_device_view, decltype(deleter)>{
      new table_device_view(source_view, stream), deleter};
  }

 private:
  table_device_view(table_view source_view, rmm::cuda_stream_view stream)
    : detail::table_device_view_base<column_device_view, table_view>(source_view, stream)
  {
  }
};

/**
 * @brief Mutable table device view that is usable in device memory
 *
 * Elements of the table can be modified in device memory.
 */
class mutable_table_device_view
  : public detail::table_device_view_base<mutable_column_device_view, mutable_table_view> {
 public:
  /**
   * @brief Factory to construct a mutable table device view that is usable in device memory.
   *
   * Allocates and copies views of `source_view`'s children to device memory to
   * make them accessible in device code.
   *
   * Returns a `std::unique_ptr<mutable_table_device_view>` with a custom deleter to
   * free the device memory allocated for the children.
   *
   * @param source_view The table view whose contents will be copied to create a new table
   * @param stream CUDA stream used for device memory operations
   * @return A `unique_ptr` to a `mutable_table_device_view` that makes the data from `source_view`
   * available in device memory
   */
  static auto create(mutable_table_view source_view,
                     rmm::cuda_stream_view stream = cudf::get_default_stream())
  {
    auto deleter = [](mutable_table_device_view* t) { t->destroy(); };
    return std::unique_ptr<mutable_table_device_view, decltype(deleter)>{
      new mutable_table_device_view(source_view, stream), deleter};
  }

 private:
  mutable_table_device_view(mutable_table_view source_view, rmm::cuda_stream_view stream)
    : detail::table_device_view_base<mutable_column_device_view, mutable_table_view>(source_view,
                                                                                     stream)
  {
  }
};

/**
 * @brief Copies the contents of a table_view to a column device view in contiguous device memory
 *
 * @tparam ColumnDeviceView The column device view type to copy to
 * @tparam HostTableView The type of the table_view to copy from
 * @param source_view The table_view to copy from
 * @param stream The stream to use for device memory allocation
 * @return tuple of device_buffer and @p ColumnDeviceView device pointer
 */
template <typename ColumnDeviceView, typename HostTableView>
auto contiguous_copy_column_device_views(HostTableView source_view, rmm::cuda_stream_view stream)
{
  // First calculate the size of memory needed to hold the
  // table's ColumnDeviceViews. This is done by calling extent()
  // for each of the table's ColumnViews columns.
  std::size_t views_size_bytes = std::accumulate(
    source_view.begin(), source_view.end(), std::size_t{0}, [](std::size_t init, auto col) {
      return init + ColumnDeviceView::extent(col);
    });
  // pad the allocation for aligning the first pointer
  auto padded_views_size_bytes = views_size_bytes + std::size_t{alignof(ColumnDeviceView) - 1};
  // A buffer of CPU memory is allocated to hold the ColumnDeviceView
  // objects. Once filled, the CPU memory is then copied to device memory
  // and the pointer is set in the d_columns member.
  auto h_buffer = cudf::detail::make_host_vector<int8_t>(padded_views_size_bytes, stream);
  // Each ColumnDeviceView instance may have child objects which may
  // require setting some internal device pointers before being copied
  // from CPU to device.
  // Allocate the device memory to be used in the result.
  // We need this pointer in order to pass it down when creating the
  // ColumnDeviceViews so the column can set the pointer(s) for any
  // of its child objects.
  // align both h_ptr, d_ptr
  auto descendant_storage = std::make_unique<rmm::device_buffer>(padded_views_size_bytes, stream);
  void* h_ptr             = detail::align_ptr_for_type<ColumnDeviceView>(h_buffer.data());
  void* d_ptr    = detail::align_ptr_for_type<ColumnDeviceView>(descendant_storage->data());
  auto d_columns = detail::child_columns_to_device_array<ColumnDeviceView>(
    source_view.begin(), source_view.end(), h_ptr, d_ptr);

  auto const h_span = host_span<int8_t const>{h_buffer}.subspan(
    static_cast<int8_t const*>(h_ptr) - h_buffer.data(), views_size_bytes);
  auto const d_span = device_span<int8_t>{static_cast<int8_t*>(d_ptr), views_size_bytes};
  cudf::detail::cuda_memcpy(d_span, h_span, stream);
  return std::make_tuple(std::move(descendant_storage), d_columns);
}

}  // namespace CUDF_EXPORT cudf
