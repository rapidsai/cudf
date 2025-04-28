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

#include "arrow_utilities.hpp"
#include "from_arrow_host.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cudf {
namespace detail {

namespace {

struct dispatch_copy_from_arrow_host {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  std::unique_ptr<rmm::device_buffer> get_mask_buffer(ArrowArray const* array)
  {
    auto* bitmap = array->buffers[validity_buffer_idx];
    if (bitmap == nullptr) { return std::make_unique<rmm::device_buffer>(0, stream, mr); }

    auto const bitmask_size = array->length + array->offset;
    auto const allocation_size =
      bitmask_allocation_size_bytes(static_cast<size_type>(bitmask_size));
    auto mask = std::make_unique<rmm::device_buffer>(allocation_size, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(mask->data(),
                                  reinterpret_cast<uint8_t const*>(bitmap),
                                  allocation_size,
                                  cudaMemcpyDefault,
                                  stream.value()));
    return mask;
  }

  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>() && !is_fixed_point<T>())>
  std::unique_ptr<column> operator()(ArrowSchemaView*, ArrowArray const*, data_type, bool)
  {
    CUDF_FAIL("Unsupported type in copy_from_arrow_host.");
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>() || is_fixed_point<T>())>
  std::unique_ptr<column> operator()(ArrowSchemaView* schema,
                                     ArrowArray const* input,
                                     data_type type,
                                     bool skip_mask)
  {
    using DeviceType = device_storage_type_t<T>;

    size_type const num_rows   = input->length;
    size_type const offset     = input->offset;
    size_type const null_count = input->null_count;
    auto data_buffer           = input->buffers[fixed_width_data_buffer_idx];

    auto const has_nulls = skip_mask ? false : input->buffers[validity_buffer_idx] != nullptr;
    auto col = make_fixed_width_column(type, num_rows, mask_state::UNALLOCATED, stream, mr);
    auto mutable_column_view = col->mutable_view();
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(mutable_column_view.data<DeviceType>(),
                      reinterpret_cast<uint8_t const*>(data_buffer) + offset * sizeof(DeviceType),
                      sizeof(DeviceType) * num_rows,
                      cudaMemcpyDefault,
                      stream.value()));

    if (has_nulls) {
      auto tmp_mask = get_mask_buffer(input);

      // if array is sliced, we have to copy the whole mask and then take copy
      auto out_mask =
        (offset == 0)
          ? std::move(*tmp_mask)
          : cudf::detail::copy_bitmask(
              static_cast<bitmask_type*>(tmp_mask->data()), offset, offset + num_rows, stream, mr);

      col->set_null_mask(std::move(out_mask), null_count);
    }

    return col;
  }
};

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<bool>(ArrowSchemaView* schema,
                                                                        ArrowArray const* input,
                                                                        data_type type,
                                                                        bool skip_mask)
{
  auto data_buffer         = input->buffers[fixed_width_data_buffer_idx];
  auto const buffer_length = bitmask_allocation_size_bytes(input->length + input->offset);

  auto data = rmm::device_buffer(buffer_length, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(data.data(),
                                reinterpret_cast<uint8_t const*>(data_buffer),
                                buffer_length,
                                cudaMemcpyDefault,
                                stream.value()));
  auto out_col = mask_to_bools(static_cast<bitmask_type*>(data.data()),
                               input->offset,
                               input->offset + input->length,
                               stream,
                               mr);

  auto const has_nulls = skip_mask ? false : input->buffers[validity_buffer_idx] != nullptr;
  if (has_nulls) {
    auto out_mask = detail::copy_bitmask(static_cast<bitmask_type*>(get_mask_buffer(input)->data()),
                                         input->offset,
                                         input->offset + input->length,
                                         stream,
                                         mr);

    out_col->set_null_mask(std::move(out_mask), input->null_count);
  }

  return out_col;
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::string_view>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  if (input->length == 0) { return make_empty_column(type_id::STRING); }
  return string_column_from_arrow_host(schema, input, get_mask_buffer(input), stream, mr);
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::dictionary32>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  ArrowSchemaView keys_schema_view;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaViewInit(&keys_schema_view, schema->schema->dictionary, nullptr));

  auto const keys_type = arrow_to_cudf_type(&keys_schema_view);
  auto keys_column =
    get_column_copy(&keys_schema_view, input->dictionary, keys_type, true, stream, mr);

  auto const dict_indices_type = [&schema]() -> data_type {
    // cudf dictionary requires a signed type for the indices
    switch (schema->storage_type) {
      case NANOARROW_TYPE_INT8: return data_type(type_id::INT8);
      case NANOARROW_TYPE_INT16: return data_type(type_id::INT16);
      case NANOARROW_TYPE_INT32: return data_type(type_id::INT32);
      case NANOARROW_TYPE_INT64: return data_type(type_id::INT64);
      default: CUDF_FAIL("Unsupported type_id for dictionary indices", cudf::data_type_error);
    }
  }();

  auto indices_column = get_column_copy(schema, input, dict_indices_type, false, stream, mr);
  // child columns shouldn't have masks and we need the mask in the main column
  auto column_contents = indices_column->release();
  indices_column       = std::make_unique<column>(dict_indices_type,
                                            static_cast<size_type>(input->length),
                                            std::move(*(column_contents.data)),
                                            rmm::device_buffer{},
                                            0);

  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(column_contents.null_mask)),
                                input->null_count);
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::struct_view>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  std::vector<std::unique_ptr<column>> child_columns;
  std::transform(
    input->children,
    input->children + input->n_children,
    schema->schema->children,
    std::back_inserter(child_columns),
    [this, input](ArrowArray const* child, ArrowSchema const* child_schema) {
      ArrowSchemaView view;
      NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
      auto type = arrow_to_cudf_type(&view);

      auto out = get_column_copy(&view, child, type, false, stream, mr);
      return input->offset == 0 && input->length == out->size()
               ? std::move(out)
               : std::make_unique<column>(
                   cudf::detail::slice(out->view(),
                                       static_cast<size_type>(input->offset),
                                       static_cast<size_type>(input->offset + input->length),
                                       stream),
                   stream,
                   mr);
    });

  auto out_mask = std::move(*(get_mask_buffer(input)));
  if (input->buffers[validity_buffer_idx] != nullptr) {
    out_mask = detail::copy_bitmask(static_cast<bitmask_type*>(out_mask.data()),
                                    input->offset,
                                    input->offset + input->length,
                                    stream,
                                    mr);
  }

  return make_structs_column(
    input->length, std::move(child_columns), input->null_count, std::move(out_mask), stream, mr);
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::list_view>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  void const* offset_buffers[2] = {nullptr, input->buffers[fixed_width_data_buffer_idx]};
  ArrowArray offsets_array      = {
         .length     = input->offset + input->length + 1,
         .null_count = 0,
         .offset     = 0,
         .n_buffers  = 2,
         .n_children = 0,
         .buffers    = offset_buffers,
  };
  nanoarrow::UniqueSchema offset_schema;
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(offset_schema.get(), NANOARROW_TYPE_INT32));

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, offset_schema.get(), nullptr));
  auto offsets_column =
    this->operator()<int32_t>(&view, &offsets_array, data_type(type_id::INT32), true);

  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema->schema->children[0], nullptr));
  auto child_type   = arrow_to_cudf_type(&view);
  auto child_column = get_column_copy(&view, input->children[0], child_type, false, stream, mr);

  auto const num_rows = offsets_column->size() - 1;
  auto out_col        = make_lists_column(num_rows,
                                   std::move(offsets_column),
                                   std::move(child_column),
                                   input->null_count,
                                   std::move(*get_mask_buffer(input)),
                                   stream,
                                   mr);

  return num_rows == input->length
           ? std::move(out_col)
           : std::make_unique<column>(
               cudf::detail::slice(out_col->view(),
                                   static_cast<size_type>(input->offset),
                                   static_cast<size_type>(input->offset + input->length),
                                   stream),
               stream,
               mr);
}

}  // namespace

std::unique_ptr<column> get_column_copy(ArrowSchemaView* schema,
                                        ArrowArray const* input,
                                        data_type type,
                                        bool skip_mask,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    input->length <= static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()),
    "Total number of rows in Arrow column exceeds the column size limit.",
    std::overflow_error);

  return type.id() != type_id::EMPTY
           ? std::move(type_dispatcher(
               type, dispatch_copy_from_arrow_host{stream, mr}, schema, input, type, skip_mask))
           : std::make_unique<column>(data_type(type_id::EMPTY),
                                      input->length,
                                      rmm::device_buffer{},
                                      rmm::device_buffer{},
                                      input->length);
}

std::unique_ptr<table> from_arrow_host(ArrowSchema const* schema,
                                       ArrowDeviceArray const* input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL",
               std::invalid_argument);
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CPU,
               "ArrowDeviceArray must have CPU device type for `from_arrow_host`",
               std::invalid_argument);

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));

  std::vector<std::unique_ptr<column>> columns;

  auto type = arrow_to_cudf_type(&view);
  CUDF_EXPECTS(type == data_type(type_id::STRUCT),
               "Must pass a struct to `from_arrow_host`",
               cudf::data_type_error);

  std::transform(input->array.children,
                 input->array.children + input->array.n_children,
                 view.schema->children,
                 std::back_inserter(columns),
                 [&stream, &mr](ArrowArray const* child, ArrowSchema const* child_schema) {
                   ArrowSchemaView view;
                   NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
                   auto type = arrow_to_cudf_type(&view);
                   return get_column_copy(&view, child, type, false, stream, mr);
                 });

  return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<column> from_arrow_host_column(ArrowSchema const* schema,
                                               ArrowDeviceArray const* input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL",
               std::invalid_argument);
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CPU,
               "ArrowDeviceArray must have CPU device type for `from_arrow_host_column`",
               std::invalid_argument);

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));

  auto type = arrow_to_cudf_type(&view);
  return get_column_copy(&view, &input->array, type, false, stream, mr);
}

std::unique_ptr<column> get_column_from_host_copy(ArrowSchemaView* schema,
                                                  ArrowArray const* input,
                                                  data_type type,
                                                  bool skip_mask,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  return get_column_copy(schema, input, type, skip_mask, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> from_arrow_host(ArrowSchema const* schema,
                                       ArrowDeviceArray const* input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow_host(schema, input, stream, mr);
}

std::unique_ptr<column> from_arrow_host_column(ArrowSchema const* schema,
                                               ArrowDeviceArray const* input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow_host_column(schema, input, stream, mr);
}

std::unique_ptr<table> from_arrow(ArrowSchema const* schema,
                                  ArrowArray const* input,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  ArrowDeviceArray const device_input = {
    .array       = *input,
    .device_id   = -1,
    .device_type = ARROW_DEVICE_CPU,
  };
  return detail::from_arrow_host(schema, &device_input, stream, mr);
}

std::unique_ptr<column> from_arrow_column(ArrowSchema const* schema,
                                          ArrowArray const* input,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  ArrowDeviceArray const device_input = {
    .array       = *input,
    .device_id   = -1,
    .device_type = ARROW_DEVICE_CPU,
  };
  return detail::from_arrow_host_column(schema, &device_input, stream, mr);
}

}  // namespace cudf
