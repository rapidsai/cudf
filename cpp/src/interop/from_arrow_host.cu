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
#include <cudf/detail/interop.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cudf {
namespace detail {

namespace {

__device__ inline bitmask_type get_mask_word(bitmask_type const* __restrict__ source,
                                             int64_t destination_word_index,
                                             int64_t source_begin_bit,
                                             int64_t source_end_bit)
{
  auto const word_index =
    destination_word_index + (source_begin_bit / size_in_bits<bitmask_type>());
  auto const curr_word = source[word_index];
  auto const end_index = (source_end_bit - 1) / size_in_bits<bitmask_type>();
  auto const next_word = (end_index > word_index) ? source[word_index + 1] : bitmask_type{0};
  auto const shift     = static_cast<bitmask_type>(source_begin_bit % size_in_bits<bitmask_type>());
  return __funnelshift_r(curr_word, next_word, shift);
}

CUDF_KERNEL void copy_shifted_bitmask(bitmask_type* __restrict__ destination,
                                      bitmask_type const* __restrict__ source,
                                      int64_t source_begin_bit,
                                      int64_t source_end_bit,
                                      size_type number_of_mask_words)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (thread_index_type destination_word_index = grid_1d::global_thread_id();
       destination_word_index < number_of_mask_words;
       destination_word_index += stride) {
    destination[destination_word_index] =
      detail::get_mask_word(source, destination_word_index, source_begin_bit, source_end_bit);
  }
}

template <typename OffsetType>
std::tuple<std::unique_ptr<column>, int64_t, int64_t> copy_offsets_column(
  ArrowSchemaView* schema,
  ArrowArray const* offsets,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  void const* offsets_buffer = offsets->buffers[fixed_width_data_buffer_idx];
  auto result =
    get_column_copy(schema, offsets, data_type(type_to_id<OffsetType>()), true, stream, mr);
  auto const offset = static_cast<OffsetType const*>(offsets_buffer)[offsets->offset];
  auto const length =
    static_cast<OffsetType const*>(offsets_buffer)[offsets->offset + offsets->length - 1] - offset;
  if (offsets->offset != 0) {
    auto begin = result->mutable_view().begin<OffsetType>();
    auto end   = begin + offsets->length;
    thrust::transform(
      rmm::exec_policy_nosync(stream), begin, end, begin, [offset] __device__(auto o) {
        return o - offset;
      });
  }
  return std::tuple{std::move(result), offset, length};
}
}  // namespace

std::tuple<std::unique_ptr<column>, int64_t, int64_t> get_offsets_column(
  ArrowSchemaView* schema,
  ArrowArray const* input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  void const* offsets_buffer     = input->buffers[fixed_width_data_buffer_idx];
  void const* offsets_buffers[2] = {nullptr, offsets_buffer};
  ArrowArray offsets_array       = {
          .length     = input->length + 1,
          .null_count = 0,
          .offset     = input->offset,
          .n_buffers  = 2,
          .n_children = 0,
          .buffers    = offsets_buffers,
  };

  if (schema->type == NANOARROW_TYPE_LIST || schema->type == NANOARROW_TYPE_STRING) {
    return copy_offsets_column<int32_t>(schema, &offsets_array, stream, mr);
  }
  if (schema->type == NANOARROW_TYPE_LARGE_STRING) {
    return copy_offsets_column<int64_t>(schema, &offsets_array, stream, mr);
  }

  CUDF_EXPECTS(schema->type == NANOARROW_TYPE_LARGE_LIST, "Unknown offsets parent type");

  // For large lists, convert 64-bit offsets to 32-bit on host with bounds checking
  int64_t const* large_offsets =
    reinterpret_cast<int64_t const*>(input->buffers[fixed_width_data_buffer_idx]) + input->offset;

  constexpr auto max_offset = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
  CUDF_EXPECTS(large_offsets[input->length] <= max_offset,
               "Large list offsets exceed 32-bit integer bounds",
               std::overflow_error);

  auto int32_offsets = std::vector<int32_t>(input->length + 1);
  auto const offset  = static_cast<int64_t const*>(offsets_buffer)[input->offset];
  auto const length =
    static_cast<int64_t const*>(offsets_buffer)[input->offset + input->length] - offset;
  std::transform(large_offsets,
                 large_offsets + int32_offsets.size(),
                 int32_offsets.begin(),
                 [offset](int64_t o) { return static_cast<int32_t>(o - offset); });

  offsets_buffers[1] = int32_offsets.data();

  auto result =
    get_column_copy(schema, &offsets_array, data_type(type_id::INT32), true, stream, mr);
  return std::tuple{std::move(result), offset, length};
}

namespace {

struct dispatch_copy_from_arrow_host {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  std::pair<std::unique_ptr<rmm::device_buffer>, size_type> get_mask_buffer(ArrowArray const* array)
  {
    auto bitmap = static_cast<uint8_t const*>(array->buffers[validity_buffer_idx]);
    if (bitmap == nullptr || array->null_count == 0) {
      return {std::make_unique<rmm::device_buffer>(0, stream, mr), 0};
    }

    constexpr auto bits_in_byte = static_cast<size_type>(size_in_bits<uint8_t>());

    auto const size         = static_cast<size_type>(array->length);
    auto const offset_index = array->offset / bits_in_byte;
    auto const mask_words   = num_bitmask_words(size);
    auto const padded_words = bitmask_allocation_size_bytes(size) / sizeof(bitmask_type);
    auto const copy_size    = cudf::util::div_rounding_up_safe(size, bits_in_byte);

    auto mask = rmm::device_uvector<bitmask_type>(padded_words, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      mask.data(), bitmap + offset_index, copy_size, cudaMemcpyDefault, stream.value()));

    auto const bit_index = static_cast<size_type>(array->offset % bits_in_byte);
    if (mask_words > 0 && bit_index > 0) {
      auto dest_mask = rmm::device_uvector<bitmask_type>(padded_words, stream, mr);
      cudf::detail::grid_1d config(mask_words, 256);
      copy_shifted_bitmask<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
        dest_mask.data(), mask.data(), bit_index, bit_index + array->length, mask_words);
      CUDF_CHECK_CUDA(stream.value());
      mask = std::move(dest_mask);
    }

    auto const null_count =
      mask_words > 0 ? cudf::detail::count_unset_bits(mask.data(), 0, size, stream) : 0;

    return {std::make_unique<rmm::device_buffer>(std::move(mask.release())), null_count};
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

    auto const num_rows = static_cast<size_type>(input->length);
    auto const data_buffer =
      static_cast<DeviceType const*>(input->buffers[fixed_width_data_buffer_idx]);

    auto col = make_fixed_width_column(type, num_rows, mask_state::UNALLOCATED, stream, mr);
    auto mutable_column_view = col->mutable_view();
    CUDF_CUDA_TRY(cudaMemcpyAsync(mutable_column_view.data<DeviceType>(),
                                  data_buffer + input->offset,
                                  sizeof(DeviceType) * num_rows,
                                  cudaMemcpyDefault,
                                  stream.value()));

    if (!skip_mask) {
      auto [mask, null_count] = get_mask_buffer(input);
      col->set_null_mask(std::move(*mask), null_count);
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
  auto data_buffer = static_cast<uint8_t const*>(input->buffers[fixed_width_data_buffer_idx]);

  auto const offset_index = input->offset / 8;  // size_in_bits<bitmask_type>();
  auto const data_words   = num_bitmask_words(input->length);

  auto data = rmm::device_uvector<bitmask_type>(data_words, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(data.data(),
                                data_buffer + offset_index,
                                (input->length + 7) / 8,
                                cudaMemcpyDefault,
                                stream.value()));

  auto const bit_index = input->offset % 8;  // size_in_bits<bitmask_type>();
  if (data_words > 0 && bit_index > 0) {
    auto dest_data = rmm::device_uvector<bitmask_type>(data_words, stream, mr);
    cudf::detail::grid_1d config(data_words, 256);
    copy_shifted_bitmask<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      dest_data.data(), data.data(), bit_index, bit_index + input->length, data_words);
    CUDF_CHECK_CUDA(stream.value());
    data = std::move(dest_data);
  }

  auto num_rows = static_cast<size_type>(input->length);
  auto out_col  = mask_to_bools(static_cast<bitmask_type*>(data.data()), 0, num_rows, stream, mr);

  if (!skip_mask) {
    auto [out_mask, null_count] = get_mask_buffer(input);
    out_col->set_null_mask(std::move(*out_mask), null_count);
  }

  return out_col;
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::string_view>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  if (input->length == 0) { return make_empty_column(type_id::STRING); }
  auto [mask, null_count] = !skip_mask
                              ? get_mask_buffer(input)
                              : std::pair{std::make_unique<rmm::device_buffer>(0, stream, mr), 0};
  return string_column_from_arrow_host(schema, input, std::move(mask), null_count, stream, mr);
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
  std::transform(input->children,
                 input->children + input->n_children,
                 schema->schema->children,
                 std::back_inserter(child_columns),
                 [this, input](ArrowArray const* child, ArrowSchema const* child_schema) {
                   ArrowSchemaView view;
                   NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
                   auto type = arrow_to_cudf_type(&view);

                   ArrowArray child_array(*child);
                   child_array.offset = input->offset;
                   child_array.length = input->length;

                   return get_column_copy(&view, &child_array, type, false, stream, mr);
                 });

  // auto [out_mask, null_count] = get_mask_buffer(input);
  auto [out_mask, null_count] =
    !skip_mask ? get_mask_buffer(input)
               : std::pair{std::make_unique<rmm::device_buffer>(0, stream, mr), 0};

  return make_structs_column(
    input->length, std::move(child_columns), null_count, std::move(*out_mask), stream, mr);
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::list_view>(
  ArrowSchemaView* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  // Initialize schema for 32-bit ints regardless of list type
  nanoarrow::UniqueSchema offset_schema;
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(offset_schema.get(), NANOARROW_TYPE_INT32));

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, offset_schema.get(), nullptr));

  CUDF_EXPECTS(
    input->length + 1 <= static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()),
    "Total number of rows in Arrow column exceeds the column size limit.",
    std::overflow_error);

  auto [offsets_column, offset, length] = get_offsets_column(schema, input, stream, mr);

  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema->schema->children[0], nullptr));
  auto child_type   = arrow_to_cudf_type(&view);
  auto child        = input->children[0];
  child->offset     = offset;
  child->length     = length;
  auto child_column = get_column_copy(&view, child, child_type, skip_mask, stream, mr);

  // auto [out_mask, null_count] = get_mask_buffer(input);
  auto [out_mask, null_count] =
    !skip_mask ? get_mask_buffer(input)
               : std::pair{std::make_unique<rmm::device_buffer>(0, stream, mr), 0};

  return make_lists_column(static_cast<size_type>(input->length),
                           std::move(offsets_column),
                           std::move(child_column),
                           null_count,
                           std::move(*out_mask),
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
    "number of rows in Arrow column exceeds the column size limit.",
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
