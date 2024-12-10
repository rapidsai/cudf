/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "arrow_utilities.hpp"
#include "decimal_conversion_utilities.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>
#include <sys/mman.h>

#include <iostream>

namespace cudf {
namespace detail {

namespace {

/*
  Enable Transparent Huge Pages (THP) for large (>4MB) allocations.
  `buf` is returned untouched.
  Enabling THP can improve performance of device-host memory transfers
  significantly, see <https://github.com/rapidsai/cudf/pull/13914>.
*/
void enable_hugepage(ArrowBuffer* buffer)
{
  if (buffer->size_bytes < (1u << 22u)) {  // Smaller than 4 MB
    return;
  }

#ifdef MADV_HUGEPAGE
  auto const pagesize = sysconf(_SC_PAGESIZE);
  void* addr          = const_cast<uint8_t*>(buffer->data);
  auto length{static_cast<std::size_t>(buffer->size_bytes)};
  if (std::align(pagesize, pagesize, addr, length)) {
    // Intentionally not checking for errors that may be returned by older kernel versions;
    // optimistically tries enabling huge pages.
    madvise(addr, length, MADV_HUGEPAGE);
  }
#endif
}

struct dispatch_to_arrow_host {
  cudf::column_view column;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  int populate_validity_bitmap(ArrowBitmap* bitmap) const
  {
    if (!column.has_nulls()) { return NANOARROW_OK; }

    NANOARROW_RETURN_NOT_OK(ArrowBitmapResize(bitmap, static_cast<int64_t>(column.size()), 0));
    enable_hugepage(&bitmap->buffer);
    CUDF_CUDA_TRY(cudaMemcpyAsync(bitmap->buffer.data,
                                  (column.offset() > 0)
                                    ? cudf::detail::copy_bitmask(column, stream, mr).data()
                                    : column.null_mask(),
                                  bitmap->buffer.size_bytes,
                                  cudaMemcpyDefault,
                                  stream.value()));
    return NANOARROW_OK;
  }

  template <typename T>
  int populate_data_buffer(device_span<T const> input, ArrowBuffer* buffer) const
  {
    NANOARROW_RETURN_NOT_OK(ArrowBufferResize(buffer, input.size_bytes(), 1));
    enable_hugepage(buffer);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      buffer->data, input.data(), input.size_bytes(), cudaMemcpyDefault, stream.value()));
    return NANOARROW_OK;
  }

  template <typename T,
            CUDF_ENABLE_IF(!is_rep_layout_compatible<T>() && !cudf::is_fixed_point<T>())>
  int operator()(ArrowArray*) const
  {
    CUDF_FAIL("Unsupported type for to_arrow_host", cudf::data_type_error);
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>() || is_fixed_point<T>())>
  int operator()(ArrowArray* out) const
  {
    nanoarrow::UniqueArray tmp;

    auto const storage_type = id_to_arrow_storage_type(column.type().id());
    NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), storage_type, column));

    NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));
    using DataType = device_storage_type_t<T>;
    NANOARROW_RETURN_NOT_OK(
      populate_data_buffer(device_span<DataType const>(column.data<DataType>(), column.size()),
                           ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));

    ArrowArrayMove(tmp.get(), out);
    return NANOARROW_OK;
  }
};

int get_column(cudf::column_view column,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr,
               ArrowArray* out);

template <>
int dispatch_to_arrow_host::operator()<bool>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_BOOL, column));

  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));
  auto bitmask = detail::bools_to_mask(column, stream, mr);
  NANOARROW_RETURN_NOT_OK(populate_data_buffer(
    device_span<uint8_t const>(reinterpret_cast<const uint8_t*>(bitmask.first->data()),
                               bitmask.first->size()),
    ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_host::operator()<cudf::string_view>(ArrowArray* out) const
{
  ArrowType nanoarrow_type = NANOARROW_TYPE_STRING;
  if (column.num_children() > 0 &&
      column.child(cudf::strings_column_view::offsets_column_index).type().id() ==
        cudf::type_id::INT64) {
    nanoarrow_type = NANOARROW_TYPE_LARGE_STRING;
  }

  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), nanoarrow_type, column));

  if (column.size() == 0) {
    // initialize the offset buffer with a single zero by convention
    if (nanoarrow_type == NANOARROW_TYPE_LARGE_STRING) {
      NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppendInt64(ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx), 0));
    } else {
      NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppendInt32(ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx), 0));
    }

    ArrowArrayMove(tmp.get(), out);
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));

  auto const scv     = cudf::strings_column_view(column);
  auto const offsets = scv.offsets();
  if (offsets.type().id() == cudf::type_id::INT64) {
    NANOARROW_RETURN_NOT_OK(populate_data_buffer(
      device_span<int64_t const>(offsets.data<int64_t>() + scv.offset(), scv.size() + 1),
      ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
  } else {
    NANOARROW_RETURN_NOT_OK(populate_data_buffer(
      device_span<int32_t const>(offsets.data<int32_t>() + scv.offset(), scv.size() + 1),
      ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
  }

  NANOARROW_RETURN_NOT_OK(
    populate_data_buffer(device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)),
                         ArrowArrayBuffer(tmp.get(), 2)));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_host::operator()<cudf::list_view>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_LIST, column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), 1));

  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));
  auto const lcv = cudf::lists_column_view(column);

  if (column.size() == 0) {
    // initialize the offsets buffer with a single zero by convention for 0 length
    NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendInt32(ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx), 0));
  } else {
    NANOARROW_RETURN_NOT_OK(
      populate_data_buffer(device_span<int32_t const>(lcv.offsets_begin(), (column.size() + 1)),
                           ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
  }

  NANOARROW_RETURN_NOT_OK(get_column(lcv.child(), stream, mr, tmp->children[0]));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_host::operator()<cudf::dictionary32>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(
    tmp.get(),
    id_to_arrow_type(column.child(cudf::dictionary_column_view::indices_column_index).type().id()),
    column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateDictionary(tmp.get()));

  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));
  auto dcv          = cudf::dictionary_column_view(column);
  auto dict_indices = dcv.get_indices_annotated();
  switch (dict_indices.type().id()) {
    case type_id::INT8:
    case type_id::UINT8:
      NANOARROW_RETURN_NOT_OK(populate_data_buffer(
        device_span<int8_t const>(dict_indices.data<int8_t>(), dict_indices.size()),
        ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
      break;
    case type_id::INT16:
    case type_id::UINT16:
      NANOARROW_RETURN_NOT_OK(populate_data_buffer(
        device_span<int16_t const>(dict_indices.data<int16_t>(), dict_indices.size()),
        ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
      break;
    case type_id::INT32:
    case type_id::UINT32:
      NANOARROW_RETURN_NOT_OK(populate_data_buffer(
        device_span<int32_t const>(dict_indices.data<int32_t>(), dict_indices.size()),
        ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
      break;
    case type_id::INT64:
    case type_id::UINT64:
      NANOARROW_RETURN_NOT_OK(populate_data_buffer(
        device_span<int64_t const>(dict_indices.data<int64_t>(), dict_indices.size()),
        ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
      break;
    default: CUDF_FAIL("unsupported type for dictionary indices");
  }

  NANOARROW_RETURN_NOT_OK(get_column(dcv.keys(), stream, mr, tmp->dictionary));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_host::operator()<cudf::struct_view>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;

  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_STRUCT, column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), column.num_children()));
  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));

  auto const scv = cudf::structs_column_view(column);

  for (size_t i = 0; i < size_t(tmp->n_children); ++i) {
    ArrowArray* child_ptr = tmp->children[i];
    auto const child      = scv.get_sliced_child(i, stream);
    NANOARROW_RETURN_NOT_OK(get_column(child, stream, mr, child_ptr));
  }

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

int get_column(cudf::column_view column,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr,
               ArrowArray* out)
{
  return column.type().id() != type_id::EMPTY
           ? type_dispatcher(column.type(), dispatch_to_arrow_host{column, stream, mr}, out)
           : initialize_array(out, NANOARROW_TYPE_NA, column);
}

unique_device_array_t create_device_array(nanoarrow::UniqueArray&& out)
{
  ArrowError err;
  if (ArrowArrayFinishBuildingDefault(out.get(), &err) != NANOARROW_OK) {
    std::cerr << err.message << std::endl;
    CUDF_FAIL("failed to build");
  }

  unique_device_array_t result(new ArrowDeviceArray, [](ArrowDeviceArray* arr) {
    if (arr->array.release != nullptr) { ArrowArrayRelease(&arr->array); }
    delete arr;
  });

  result->device_id   = -1;
  result->device_type = ARROW_DEVICE_CPU;
  result->sync_event  = nullptr;
  ArrowArrayMove(out.get(), &result->array);
  return result;
}

}  // namespace

unique_device_array_t to_arrow_host(cudf::table_view const& table,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_STRUCT));

  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), table.num_columns()));
  tmp->length     = table.num_rows();
  tmp->null_count = 0;

  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    auto child = tmp->children[i];
    auto col   = table.column(i);
    NANOARROW_THROW_NOT_OK(
      cudf::type_dispatcher(col.type(), detail::dispatch_to_arrow_host{col, stream, mr}, child));
  }

  // wait for all the stream operations to complete before we return.
  // this ensures that the host memory that we're returning will be populated
  // before we return from this function.
  stream.synchronize();

  return create_device_array(std::move(tmp));
}

unique_device_array_t to_arrow_host(cudf::column_view const& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray tmp;

  NANOARROW_THROW_NOT_OK(
    cudf::type_dispatcher(col.type(), detail::dispatch_to_arrow_host{col, stream, mr}, tmp.get()));

  // wait for all the stream operations to complete before we return.
  // this ensures that the host memory that we're returning will be populated
  // before we return from this function.
  stream.synchronize();

  return create_device_array(std::move(tmp));
}

}  // namespace detail

unique_device_array_t to_arrow_host(cudf::column_view const& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_host(col, stream, mr);
}

unique_device_array_t to_arrow_host(cudf::table_view const& table,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_host(table, stream, mr);
}

}  // namespace cudf
