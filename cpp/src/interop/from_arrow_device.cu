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
#include <cudf/interop/detail/arrow.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>

#include <iostream>

namespace cudf {

namespace detail {
data_type arrow_to_cudf_type(const ArrowSchemaView* arrow_view)
{
  switch (arrow_view->type) {
    case NANOARROW_TYPE_NA: return data_type(type_id::EMPTY);
    case NANOARROW_TYPE_BOOL: return data_type(type_id::BOOL8);
    case NANOARROW_TYPE_INT8: return data_type(type_id::INT8);
    case NANOARROW_TYPE_INT16: return data_type(type_id::INT16);
    case NANOARROW_TYPE_INT32: return data_type(type_id::INT32);
    case NANOARROW_TYPE_INT64: return data_type(type_id::INT64);
    case NANOARROW_TYPE_UINT8: return data_type(type_id::UINT8);
    case NANOARROW_TYPE_UINT16: return data_type(type_id::UINT16);
    case NANOARROW_TYPE_UINT32: return data_type(type_id::UINT32);
    case NANOARROW_TYPE_UINT64: return data_type(type_id::UINT64);
    case NANOARROW_TYPE_FLOAT: return data_type(type_id::FLOAT32);
    case NANOARROW_TYPE_DOUBLE: return data_type(type_id::FLOAT64);
    case NANOARROW_TYPE_DATE32: return data_type(type_id::TIMESTAMP_DAYS);
    case NANOARROW_TYPE_STRING: return data_type(type_id::STRING);
    case NANOARROW_TYPE_LIST: return data_type(type_id::LIST);
    case NANOARROW_TYPE_DICTIONARY: return data_type(type_id::DICTIONARY32);
    case NANOARROW_TYPE_STRUCT: return data_type(type_id::STRUCT);
    case NANOARROW_TYPE_TIMESTAMP: {
      switch (arrow_view->time_unit) {
        case NANOARROW_TIME_UNIT_SECOND: return data_type(type_id::TIMESTAMP_SECONDS);
        case NANOARROW_TIME_UNIT_MILLI: return data_type(type_id::TIMESTAMP_MILLISECONDS);
        case NANOARROW_TIME_UNIT_MICRO: return data_type(type_id::TIMESTAMP_MICROSECONDS);
        case NANOARROW_TIME_UNIT_NANO: return data_type(type_id::TIMESTAMP_NANOSECONDS);
        default: CUDF_FAIL("Unsupported timestamp unit in arrow", cudf::data_type_error);
      }
    }
    case NANOARROW_TYPE_DURATION: {
      switch (arrow_view->time_unit) {
        case NANOARROW_TIME_UNIT_SECOND: return data_type(type_id::DURATION_SECONDS);
        case NANOARROW_TIME_UNIT_MILLI: return data_type(type_id::DURATION_MILLISECONDS);
        case NANOARROW_TIME_UNIT_MICRO: return data_type(type_id::DURATION_MICROSECONDS);
        case NANOARROW_TIME_UNIT_NANO: return data_type(type_id::DURATION_NANOSECONDS);
        default: CUDF_FAIL("Unsupported duration unit in arrow", cudf::data_type_error);
      }
    }
    case NANOARROW_TYPE_DECIMAL128:
      return data_type{type_id::DECIMAL128, -arrow_view->decimal_scale};
    default: CUDF_FAIL("Unsupported type_id conversion to cudf", cudf::data_type_error);
  }
}

namespace {

struct dispatch_copy_from_arrow_device {
  rmm::cuda_stream_view stream;
  rmm::mr::device_memory_resource* mr;

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

  template <typename T,
            CUDF_ENABLE_IF(not is_rep_layout_compatible<T>() &&
                           !std::is_same_v<T, numeric::decimal128>)>
  std::unique_ptr<column> operator()(ArrowSchemaView*, ArrowArray const*, data_type, bool)
  {
    CUDF_FAIL("Unsupported type in copy_from_arrow_device.");
  }

  template <typename T,
            CUDF_ENABLE_IF(is_rep_layout_compatible<T>() || std::is_same_v<T, numeric::decimal128>)>
  std::unique_ptr<column> operator()(ArrowSchemaView* schema,
                                     ArrowArray const* input,
                                     data_type type,
                                     bool skip_mask)
  {
    using DeviceType = std::conditional_t<std::is_same_v<T, numeric::decimal128>, __int128_t, T>;

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

// forward declaration is needed because `type_dispatch` instantiates the
// dispatch_from_arrow_device struct causing a recursive situation for struct,
// dictionary and list_view types.
std::unique_ptr<column> get_column_copy(ArrowSchemaView* schema,
                                        ArrowArray const* input,
                                        data_type type,
                                        bool skip_mask,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr);

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_device::operator()<bool>(ArrowSchemaView* schema,
                                                                          ArrowArray const* input,
                                                                          data_type type,
                                                                          bool skip_mask)
{
  auto data_buffer         = input->buffers[fixed_width_data_buffer_idx];
  const auto buffer_length = bitmask_allocation_size_bytes(input->length + input->offset);
  // mask-to-bools expects the mask to be bitmask_type aligned/padded
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
std::unique_ptr<column> dispatch_copy_from_arrow_device::operator()<cudf::string_view>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  if (input->length == 0) { return make_empty_column(type_id::STRING); }

  const void* offset_buffers[2] = {nullptr, input->buffers[fixed_width_data_buffer_idx]};
  ArrowArray offsets_array      = {
         .length     = input->offset + input->length + 1,
         .null_count = 0,
         .offset     = 0,
         .n_buffers  = 2,
         .n_children = 0,
         .buffers    = offset_buffers,
  };

  size_type const char_data_length =
    reinterpret_cast<int32_t const*>(offset_buffers[1])[input->length + input->offset];
  const void* char_buffers[2] = {nullptr, input->buffers[2]};
  ArrowArray char_array       = {
          .length     = char_data_length,
          .null_count = 0,
          .offset     = 0,
          .n_buffers  = 2,
          .n_children = 0,
          .buffers    = char_buffers,
  };

  nanoarrow::UniqueSchema offset_schema;
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(offset_schema.get(), NANOARROW_TYPE_INT32));

  nanoarrow::UniqueSchema char_data_schema;
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(char_data_schema.get(), NANOARROW_TYPE_INT8));

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, offset_schema.get(), nullptr));
  auto offsets_column =
    this->operator()<int32_t>(&view, &offsets_array, data_type(type_id::INT32), true);
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, char_data_schema.get(), nullptr));
  auto chars_column = this->operator()<int8_t>(&view, &char_array, data_type(type_id::INT8), true);

  auto const num_rows = offsets_column->size() - 1;
  auto out_col        = make_strings_column(num_rows,
                                     std::move(offsets_column),
                                     std::move(chars_column->release().data.release()[0]),
                                     input->null_count,
                                     std::move(*get_mask_buffer(input)));

  return input->offset == 0
           ? std::move(out_col)
           : std::make_unique<column>(
               cudf::detail::slice(out_col,
                                   static_cast<size_type>(input->offset),
                                   static_cast<size_type>(input->offset + input->length),
                                   stream),
               stream,
               mr);
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_device::operator()<cudf::dictionary32>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  ArrowSchemaView keys_schema_view;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaViewInit(&keys_schema_view, schema->schema->dictionary, nullptr));

  auto const keys_type = arrow_to_cudf_type(&keys_schema_view);
  auto keys_column =
    get_column_copy(&keys_schema_view, input->dictionary, keys_type, true, stream, mr);

  auto const dict_indices_type = [&schema]() -> data_type {
    // cudf dictionary requires an unsigned type for the indices,
    // since it is invalid for an arrow dictionary to contain negative
    // indices, we can safely use the unsigned equivalent without having
    // to modify the buffers.
    switch (schema->storage_type) {
      case NANOARROW_TYPE_INT8:
      case NANOARROW_TYPE_UINT8: return data_type(type_id::UINT8);
      case NANOARROW_TYPE_INT16:
      case NANOARROW_TYPE_UINT16: return data_type(type_id::UINT16);
      case NANOARROW_TYPE_INT32:
      case NANOARROW_TYPE_UINT32: return data_type(type_id::UINT32);
      case NANOARROW_TYPE_INT64:
      case NANOARROW_TYPE_UINT64: return data_type(type_id::UINT64);
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
std::unique_ptr<column> dispatch_copy_from_arrow_device::operator()<cudf::struct_view>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  char buffer[1024];

  std::vector<std::unique_ptr<column>> child_columns;
  std::transform(input->children,
                 input->children + input->n_children,
                 schema->schema->children,
                 std::back_inserter(child_columns),
                 [this, input, &buffer](ArrowArray const* child, ArrowSchema const* child_schema) {
                   ArrowSchemaView view;
                   NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
                   auto type = arrow_to_cudf_type(&view);

                   auto out = get_column_copy(&view, child, type, false, stream, mr);
                   return std::make_unique<column>(
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
std::unique_ptr<column> dispatch_copy_from_arrow_device::operator()<cudf::list_view>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  const void* offset_buffers[2] = {nullptr, input->buffers[fixed_width_data_buffer_idx]};
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

using dispatch_tuple_t = std::tuple<column_view, owned_columns_t>;

struct dispatch_from_arrow_device {
  template <typename T,
            CUDF_ENABLE_IF(not is_rep_layout_compatible<T>() &&
                           !std::is_same_v<T, numeric::decimal128>)>
  dispatch_tuple_t operator()(ArrowSchemaView*,
                              ArrowArray const*,
                              data_type,
                              bool,
                              rmm::cuda_stream_view,
                              rmm::mr::device_memory_resource*)
  {
    CUDF_FAIL("Unsupported type in from_arrow_device", cudf::data_type_error);
  }

  template <typename T,
            CUDF_ENABLE_IF(is_rep_layout_compatible<T>() || std::is_same_v<T, numeric::decimal128>)>
  dispatch_tuple_t operator()(ArrowSchemaView* schema,
                              ArrowArray const* input,
                              data_type type,
                              bool skip_mask,
                              rmm::cuda_stream_view,
                              rmm::mr::device_memory_resource*)
  {
    size_type const num_rows   = input->length;
    size_type const offset     = input->offset;
    size_type const null_count = input->null_count;
    bitmask_type const* null_mask =
      skip_mask ? nullptr
                : reinterpret_cast<bitmask_type const*>(input->buffers[validity_buffer_idx]);
    auto data_buffer = input->buffers[fixed_width_data_buffer_idx];
    return std::make_tuple<column_view, owned_columns_t>(
      {type, num_rows, data_buffer, null_mask, null_count, offset}, {});
  }
};

// forward declaration is needed because `type_dispatch` instantiates the
// dispatch_from_arrow_device struct causing a recursive situation for struct,
// dictionary and list_view types.
dispatch_tuple_t get_column(ArrowSchemaView* schema,
                            ArrowArray const* input,
                            data_type type,
                            bool skip_mask,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr);

template <>
dispatch_tuple_t dispatch_from_arrow_device::operator()<bool>(ArrowSchemaView* schema,
                                                              ArrowArray const* input,
                                                              data_type type,
                                                              bool skip_mask,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::mr::device_memory_resource* mr)
{
  if (input->length == 0) {
    return std::make_tuple<column_view, owned_columns_t>(
      {type,
       0,
       nullptr,
       skip_mask ? nullptr
                 : reinterpret_cast<bitmask_type const*>(input->buffers[validity_buffer_idx]),
       0},
      {});
  }

  auto out_col = mask_to_bools(
    reinterpret_cast<bitmask_type const*>(input->buffers[fixed_width_data_buffer_idx]),
    input->offset,
    input->offset + input->length,
    stream,
    mr);
  auto const has_nulls = skip_mask ? false : input->buffers[validity_buffer_idx] != nullptr;
  if (has_nulls) {
    auto out_mask = cudf::detail::copy_bitmask(
      reinterpret_cast<bitmask_type const*>(input->buffers[validity_buffer_idx]),
      input->offset,
      input->offset + input->length,
      stream,
      mr);
    out_col->set_null_mask(std::move(out_mask), input->null_count);
  }

  auto out_view = out_col->view();
  owned_columns_t owned;
  owned.emplace_back(std::move(out_col));
  return std::make_tuple<column_view, owned_columns_t>(std::move(out_view), std::move(owned));
}

template <>
dispatch_tuple_t dispatch_from_arrow_device::operator()<cudf::string_view>(
  ArrowSchemaView* schema,
  ArrowArray const* input,
  data_type type,
  bool skip_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  if (input->length == 0) {
    return std::make_tuple<column_view, owned_columns_t>(
      {type,
       0,
       nullptr,
       skip_mask ? nullptr
                 : reinterpret_cast<bitmask_type const*>(input->buffers[validity_buffer_idx]),
       0},
      {});
  }

  auto offsets_view = column_view{data_type(type_id::INT32),
                                  static_cast<size_type>(input->offset + input->length) + 1,
                                  input->buffers[fixed_width_data_buffer_idx],
                                  nullptr,
                                  0,
                                  0};
  return std::make_tuple<column_view, owned_columns_t>(
    {type,
     static_cast<size_type>(input->length),
     input->buffers[2],
     skip_mask ? nullptr
               : reinterpret_cast<bitmask_type const*>(input->buffers[validity_buffer_idx]),
     static_cast<size_type>(input->null_count),
     static_cast<size_type>(input->offset),
     {offsets_view}},
    {});
}

template <>
dispatch_tuple_t dispatch_from_arrow_device::operator()<cudf::dictionary32>(
  ArrowSchemaView* schema,
  ArrowArray const* input,
  data_type type,
  bool skip_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  ArrowSchemaView keys_schema_view;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaViewInit(&keys_schema_view, schema->schema->dictionary, nullptr));

  auto const keys_type = arrow_to_cudf_type(&keys_schema_view);
  auto [keys_view, owned_cols] =
    get_column(&keys_schema_view, input->dictionary, keys_type, true, stream, mr);

  auto const dict_indices_type = [&schema]() -> data_type {
    // cudf dictionary requires an unsigned type for the indices,
    // since it is invalid for an arrow dictionary to contain negative
    // indices, we can safely use the unsigned equivalent without having
    // to modify the buffers.
    switch (schema->storage_type) {
      case NANOARROW_TYPE_INT8:
      case NANOARROW_TYPE_UINT8: return data_type(type_id::UINT8);
      case NANOARROW_TYPE_INT16:
      case NANOARROW_TYPE_UINT16: return data_type(type_id::UINT16);
      case NANOARROW_TYPE_INT32:
      case NANOARROW_TYPE_UINT32: return data_type(type_id::UINT32);
      case NANOARROW_TYPE_INT64:
      case NANOARROW_TYPE_UINT64: return data_type(type_id::UINT64);
      default: CUDF_FAIL("Unsupported type_id for dictionary indices", cudf::data_type_error);
    }
  }();

  size_type const num_rows   = input->length;
  size_type const offset     = input->offset;
  size_type const null_count = input->null_count;
  column_view indices_view   = column_view{dict_indices_type,
                                         offset + num_rows,
                                         input->buffers[fixed_width_data_buffer_idx],
                                         nullptr,
                                         0,
                                         0};

  return std::make_tuple<column_view, owned_columns_t>(
    {type,
     num_rows,
     nullptr,
     reinterpret_cast<bitmask_type const*>(input->buffers[validity_buffer_idx]),
     null_count,
     offset,
     {indices_view, keys_view}},
    std::move(owned_cols));
}

template <>
dispatch_tuple_t dispatch_from_arrow_device::operator()<cudf::struct_view>(
  ArrowSchemaView* schema,
  ArrowArray const* input,
  data_type type,
  bool skip_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  std::vector<column_view> children;
  owned_columns_t out_owned_cols;
  std::transform(
    input->children,
    input->children + input->n_children,
    schema->schema->children,
    std::back_inserter(children),
    [&out_owned_cols, &stream, &mr](ArrowArray const* child, ArrowSchema const* child_schema) {
      ArrowSchemaView view;
      NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
      auto type              = arrow_to_cudf_type(&view);
      auto [out_view, owned] = get_column(&view, child, type, false, stream, mr);
      if (out_owned_cols.empty()) {
        out_owned_cols = std::move(owned);
      } else {
        out_owned_cols.insert(std::end(out_owned_cols),
                              std::make_move_iterator(std::begin(owned)),
                              std::make_move_iterator(std::end(owned)));
      }
      return out_view;
    });

  size_type const num_rows   = input->length;
  size_type const offset     = input->offset;
  size_type const null_count = input->null_count;
  return std::make_tuple<column_view, owned_columns_t>(
    {type,
     num_rows,
     nullptr,
     reinterpret_cast<bitmask_type const*>(input->buffers[validity_buffer_idx]),
     null_count,
     offset,
     std::move(children)},
    std::move(out_owned_cols));
}

template <>
dispatch_tuple_t dispatch_from_arrow_device::operator()<cudf::list_view>(
  ArrowSchemaView* schema,
  ArrowArray const* input,
  data_type type,
  bool skip_mask,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  size_type const num_rows   = input->length;
  size_type const offset     = input->offset;
  size_type const null_count = input->null_count;
  auto offsets_view          = column_view{data_type(type_id::INT32),
                                  offset + num_rows + 1,
                                  input->buffers[fixed_width_data_buffer_idx],
                                  nullptr,
                                  0,
                                  0};

  ArrowSchemaView child_schema_view;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaViewInit(&child_schema_view, schema->schema->children[0], nullptr));
  auto child_type = arrow_to_cudf_type(&child_schema_view);
  auto [child_view, owned] =
    get_column(&child_schema_view, input->children[0], child_type, false, stream, mr);

  // in the scenario where we were sliced and there are more elements in the child_view
  // than can be referenced by the sliced offsets, we need to slice the child_view
  // so that when `get_sliced_child` is called, we still produce the right result
  auto max_child_offset = cudf::detail::get_value<int32_t>(offsets_view, offset + num_rows, stream);
  child_view            = cudf::slice(child_view, {0, max_child_offset}, stream).front();

  return std::make_tuple<column_view, owned_columns_t>(
    {type,
     num_rows,
     rmm::device_buffer{0, stream, mr}.data(),
     reinterpret_cast<bitmask_type const*>(input->buffers[validity_buffer_idx]),
     null_count,
     offset,
     {offsets_view, child_view}},
    std::move(owned));
}

std::unique_ptr<column> get_column_copy(ArrowSchemaView* schema,
                                        ArrowArray const* input,
                                        data_type type,
                                        bool skip_mask,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  return type.id() != type_id::EMPTY
           ? std::move(type_dispatcher(
               type, dispatch_copy_from_arrow_device{stream, mr}, schema, input, type, skip_mask))
           : std::make_unique<column>(data_type(type_id::EMPTY),
                                      input->length,
                                      rmm::device_buffer{},
                                      rmm::device_buffer{},
                                      input->length);
}

dispatch_tuple_t get_column(ArrowSchemaView* schema,
                            ArrowArray const* input,
                            data_type type,
                            bool skip_mask,
                            rmm::cuda_stream_view stream,
                            rmm::mr::device_memory_resource* mr)
{
  return type.id() != type_id::EMPTY
           ? std::move(type_dispatcher(
               type, dispatch_from_arrow_device{}, schema, input, type, skip_mask, stream, mr))
           : std::make_tuple<column_view, owned_columns_t>({data_type(type_id::EMPTY),
                                                            static_cast<size_type>(input->length),
                                                            nullptr,
                                                            nullptr,
                                                            static_cast<size_type>(input->length)},
                                                           {});
}

}  // namespace

unique_table_view_t from_arrow_device(ArrowSchemaView* schema,
                                      ArrowDeviceArray const* input,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  if (input->sync_event != nullptr) {
    CUDF_CUDA_TRY(
      cudaStreamWaitEvent(stream.value(), *reinterpret_cast<cudaEvent_t*>(input->sync_event)));
  }

  std::vector<column_view> columns;
  owned_columns_t owned_mem;

  auto type = arrow_to_cudf_type(schema);
  CUDF_EXPECTS(type == data_type(type_id::STRUCT),
               "Must pass a struct to `from_arrow_device`",
               cudf::data_type_error);
  std::transform(
    input->array.children,
    input->array.children + input->array.n_children,
    schema->schema->children,
    std::back_inserter(columns),
    [&owned_mem, &stream, &mr](ArrowArray const* child, ArrowSchema const* child_schema) {
      ArrowSchemaView view;
      NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
      auto type              = arrow_to_cudf_type(&view);
      auto [out_view, owned] = get_column(&view, child, type, false, stream, mr);
      if (owned_mem.empty()) {
        owned_mem = std::move(owned);
      } else {
        owned_mem.insert(std::end(owned_mem),
                         std::make_move_iterator(std::begin(owned)),
                         std::make_move_iterator(std::end(owned)));
      }
      return out_view;
    });

  return unique_table_view_t{new table_view{columns},
                             custom_view_deleter<cudf::table_view>{std::move(owned_mem)}};
}

unique_column_view_t from_arrow_device_column(ArrowSchemaView* schema,
                                              ArrowDeviceArray const* input,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  if (input->sync_event != nullptr) {
    CUDF_CUDA_TRY(
      cudaStreamWaitEvent(stream.value(), *reinterpret_cast<cudaEvent_t*>(input->sync_event)));
  }

  auto type             = arrow_to_cudf_type(schema);
  auto [colview, owned] = get_column(schema, &input->array, type, false, stream, mr);
  return unique_column_view_t{new column_view{colview},
                              custom_view_deleter<cudf::column_view>{std::move(owned)}};
}

std::unique_ptr<table> from_arrow_host(ArrowSchemaView* schema,
                                       ArrowDeviceArray const* input,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  std::vector<std::unique_ptr<column>> columns;

  auto type = arrow_to_cudf_type(schema);
  CUDF_EXPECTS(type == data_type(type_id::STRUCT),
               "Must pass a struct to `from_arrow_host`",
               cudf::data_type_error);

  std::transform(input->array.children,
                 input->array.children + input->array.n_children,
                 schema->schema->children,
                 std::back_inserter(columns),
                 [&stream, &mr](ArrowArray const* child, ArrowSchema const* child_schema) {
                   ArrowSchemaView view;
                   NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
                   auto type = arrow_to_cudf_type(&view);
                   return get_column_copy(&view, child, type, false, stream, mr);
                 });

  return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<column> from_arrow_host_column(ArrowSchemaView* schema,
                                               ArrowDeviceArray const* input,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  auto type = arrow_to_cudf_type(schema);
  return get_column_copy(schema, &input->array, type, false, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> from_arrow_host(ArrowSchema const* schema,
                                       ArrowDeviceArray const* input,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL");
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CPU,
               "ArrowDeviceArray must have CPU device type for `from_arrow_host`");

  CUDF_FUNC_RANGE();

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));
  return detail::from_arrow_host(&view, input, stream, mr);
}

std::unique_ptr<column> from_arrow_host_column(ArrowSchema const* schema,
                                               ArrowDeviceArray const* input,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL");
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CPU,
               "ArrowDeviceArray must have CPU device type for `from_arrow_host_column`");

  CUDF_FUNC_RANGE();

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));
  return detail::from_arrow_host_column(&view, input, stream, mr);
}

unique_table_view_t from_arrow_device(ArrowSchema const* schema,
                                      ArrowDeviceArray const* input,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL");
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CUDA ||
                 input->device_type == ARROW_DEVICE_CUDA_HOST ||
                 input->device_type == ARROW_DEVICE_CUDA_MANAGED,
               "ArrowDeviceArray memory must be accessible to CUDA");

  CUDF_FUNC_RANGE();

  rmm::cuda_set_device_raii dev(
    rmm::cuda_device_id{static_cast<rmm::cuda_device_id::value_type>(input->device_id)});
  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));
  return detail::from_arrow_device(&view, input, stream, mr);
}

unique_column_view_t from_arrow_device_column(ArrowSchema const* schema,
                                              ArrowDeviceArray const* input,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL");
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CUDA ||
                 input->device_type == ARROW_DEVICE_CUDA_HOST ||
                 input->device_type == ARROW_DEVICE_CUDA_MANAGED,
               "ArrowDeviceArray must be accessible to CUDA");

  CUDF_FUNC_RANGE();

  rmm::cuda_set_device_raii dev(
    rmm::cuda_device_id{static_cast<rmm::cuda_device_id::value_type>(input->device_id)});
  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));
  return detail::from_arrow_device_column(&view, input, stream, mr);
}

}  // namespace cudf
