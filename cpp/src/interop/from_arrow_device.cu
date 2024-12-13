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

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/unary.hpp>
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

namespace cudf {

namespace detail {

namespace {

using dispatch_tuple_t = std::tuple<column_view, owned_columns_t>;

struct dispatch_from_arrow_device {
  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>() && !is_fixed_point<T>())>
  dispatch_tuple_t operator()(ArrowSchemaView*,
                              ArrowArray const*,
                              data_type,
                              bool,
                              rmm::cuda_stream_view,
                              rmm::device_async_resource_ref)
  {
    CUDF_FAIL("Unsupported type in from_arrow_device", cudf::data_type_error);
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>() || is_fixed_point<T>())>
  dispatch_tuple_t operator()(ArrowSchemaView* schema,
                              ArrowArray const* input,
                              data_type type,
                              bool skip_mask,
                              rmm::cuda_stream_view,
                              rmm::device_async_resource_ref mr)
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
                            rmm::device_async_resource_ref mr);

template <>
dispatch_tuple_t dispatch_from_arrow_device::operator()<bool>(ArrowSchemaView* schema,
                                                              ArrowArray const* input,
                                                              data_type type,
                                                              bool skip_mask,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
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
  rmm::device_async_resource_ref mr)
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

  data_type offsets_type(type_id::INT32);
  if (schema->type == NANOARROW_TYPE_LARGE_STRING) { offsets_type = data_type(type_id::INT64); }
  auto offsets_view = column_view{offsets_type,
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
  rmm::device_async_resource_ref mr)
{
  ArrowSchemaView keys_schema_view;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaViewInit(&keys_schema_view, schema->schema->dictionary, nullptr));

  auto const keys_type = arrow_to_cudf_type(&keys_schema_view);
  auto [keys_view, owned_cols] =
    get_column(&keys_schema_view, input->dictionary, keys_type, true, stream, mr);

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
  rmm::device_async_resource_ref mr)
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
  rmm::device_async_resource_ref mr)
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

dispatch_tuple_t get_column(ArrowSchemaView* schema,
                            ArrowArray const* input,
                            data_type type,
                            bool skip_mask,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
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

unique_table_view_t from_arrow_device(ArrowSchema const* schema,
                                      ArrowDeviceArray const* input,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL",
               std::invalid_argument);
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CUDA ||
                 input->device_type == ARROW_DEVICE_CUDA_HOST ||
                 input->device_type == ARROW_DEVICE_CUDA_MANAGED,
               "ArrowDeviceArray memory must be accessible to CUDA",
               std::invalid_argument);

  rmm::cuda_set_device_raii dev(
    rmm::cuda_device_id{static_cast<rmm::cuda_device_id::value_type>(input->device_id)});
  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));

  if (input->sync_event != nullptr) {
    CUDF_CUDA_TRY(
      cudaStreamWaitEvent(stream.value(), *reinterpret_cast<cudaEvent_t*>(input->sync_event)));
  }

  std::vector<column_view> columns;
  owned_columns_t owned_mem;

  auto type = arrow_to_cudf_type(&view);
  CUDF_EXPECTS(type == data_type(type_id::STRUCT),
               "Must pass a struct to `from_arrow_device`",
               cudf::data_type_error);
  std::transform(
    input->array.children,
    input->array.children + input->array.n_children,
    view.schema->children,
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

unique_column_view_t from_arrow_device_column(ArrowSchema const* schema,
                                              ArrowDeviceArray const* input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL",
               std::invalid_argument);
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CUDA ||
                 input->device_type == ARROW_DEVICE_CUDA_HOST ||
                 input->device_type == ARROW_DEVICE_CUDA_MANAGED,
               "ArrowDeviceArray must be accessible to CUDA",
               std::invalid_argument);

  rmm::cuda_set_device_raii dev(
    rmm::cuda_device_id{static_cast<rmm::cuda_device_id::value_type>(input->device_id)});
  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));

  if (input->sync_event != nullptr) {
    CUDF_CUDA_TRY(
      cudaStreamWaitEvent(stream.value(), *reinterpret_cast<cudaEvent_t*>(input->sync_event)));
  }

  auto type             = arrow_to_cudf_type(&view);
  auto [colview, owned] = get_column(&view, &input->array, type, false, stream, mr);
  return unique_column_view_t{new column_view{colview},
                              custom_view_deleter<cudf::column_view>{std::move(owned)}};
}

}  // namespace detail

unique_table_view_t from_arrow_device(ArrowSchema const* schema,
                                      ArrowDeviceArray const* input,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow_device(schema, input, stream, mr);
}

unique_column_view_t from_arrow_device_column(ArrowSchema const* schema,
                                              ArrowDeviceArray const* input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow_device_column(schema, input, stream, mr);
}

}  // namespace cudf
