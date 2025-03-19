/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/transform.h>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>

#include <numeric>
#include <string>
#include <vector>

namespace cudf {
namespace detail {

namespace {

constexpr int chars_buffer_idx = 2;

std::unique_ptr<column> from_arrow_string(ArrowSchemaView* schema,
                                          ArrowArray const* input,
                                          std::unique_ptr<rmm::device_buffer>&& mask,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  // offsets column should contain no nulls so we can put nullptr for the bitmask
  // nulls are tracked in the parent string column itself, not in the offsets
  void const* offset_buffers[] = {nullptr, input->buffers[fixed_width_data_buffer_idx]};
  ArrowArray offsets_array     = {
        .length     = input->offset + input->length + 1,
        .null_count = 0,
        .offset     = 0,
        .n_buffers  = 2,
        .n_children = 0,
        .buffers    = offset_buffers,
  };

  // chars_column does not contain any nulls, they are tracked by the parent string column
  // itself instead. So we pass nullptr for the validity bitmask.
  int64_t const char_data_length = [&]() {
    if (schema->type == NANOARROW_TYPE_LARGE_STRING) {
      return reinterpret_cast<int64_t const*>(offset_buffers[1])[input->length + input->offset];
    } else if (schema->type == NANOARROW_TYPE_STRING) {
      return static_cast<int64_t>(
        reinterpret_cast<int32_t const*>(offset_buffers[1])[input->length + input->offset]);
    } else {
      CUDF_FAIL("Unsupported string type", cudf::data_type_error);
    }
  }();
  void const* char_buffers[] = {nullptr, input->buffers[chars_buffer_idx]};
  ArrowArray char_array      = {
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

  // leverage the dispatch overloads for int32 and char(int8) to generate the child
  // offset and char data columns for us.
  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, offset_schema.get(), nullptr));
  auto offsets_column = [&]() {
    if (schema->type == NANOARROW_TYPE_LARGE_STRING) {
      return get_column_copy(&view, &offsets_array, data_type(type_id::INT64), true, stream, mr);
    } else if (schema->type == NANOARROW_TYPE_STRING) {
      return get_column_copy(&view, &offsets_array, data_type(type_id::INT32), true, stream, mr);
    } else {
      CUDF_FAIL("Unsupported string type", cudf::data_type_error);
    }
  }();
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, char_data_schema.get(), nullptr));

  rmm::device_buffer chars(char_data_length, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(chars.data(),
                                reinterpret_cast<uint8_t const*>(char_array.buffers[1]),
                                chars.size(),
                                cudaMemcpyDefault,
                                stream.value()));
  auto const num_rows = offsets_column->size() - 1;
  auto out_col        = make_strings_column(num_rows,
                                     std::move(offsets_column),
                                     std::move(chars),
                                     input->null_count,
                                     std::move(*mask.release()));

  return input->offset == 0
           ? std::move(out_col)
           : std::make_unique<column>(
               cudf::detail::slice(out_col->view(),
                                   static_cast<size_type>(input->offset),
                                   static_cast<size_type>(input->offset + input->length),
                                   stream),
               stream,
               mr);
}

constexpr int stringview_vector_idx = 1;

std::unique_ptr<column> from_arrow_stringview(ArrowSchemaView* schema,
                                              ArrowArray const* input,
                                              std::unique_ptr<rmm::device_buffer>&& mask,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  ArrowArrayView view;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(&view, schema->schema, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&view, input, nullptr));

  // first copy stringview array to device
  auto items   = view.buffer_views[stringview_vector_idx].data.as_binary_view;
  auto d_items = rmm::device_uvector<ArrowBinaryView>(input->length, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_items.data(),
                                items + input->offset,
                                input->length * sizeof(ArrowBinaryView),
                                cudaMemcpyDefault,
                                stream.value()));

  // then copy variadic buffers to device
  auto variadics     = std::vector<rmm::device_buffer>();
  auto variadic_ptrs = std::vector<char const*>();
  for (auto i = 0L; i < view.n_variadic_buffers; ++i) {
    variadics.emplace_back(view.variadic_buffers[i], view.variadic_buffer_sizes[i], stream);
    variadic_ptrs.push_back(static_cast<char const*>(variadics.back().data()));
  }

  // copy variadic device pointers to device
  auto d_variadic_ptrs = cudf::detail::make_device_uvector_async(
    variadic_ptrs, stream, cudf::get_current_device_resource_ref());
  auto d_ptrs = d_variadic_ptrs.data();
  auto d_mask = static_cast<cudf::bitmask_type*>(mask->data());

  using string_index_pair = cudf::strings::detail::string_index_pair;

  // create indices to string fragments for the make_strings_column gather
  auto d_indices = rmm::device_uvector<string_index_pair>(input->length, stream, mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(input->length),
    d_indices.begin(),
    [d_items = d_items.data(), d_ptrs, d_mask] __device__(auto idx) -> string_index_pair {
      if (d_mask && !cudf::bit_is_set(d_mask, idx)) { return string_index_pair{nullptr, 0}; }
      auto const& item = d_items[idx];
      auto const size  = static_cast<cudf::size_type>(item.inlined.size);
      auto const data  = (size <= NANOARROW_BINARY_VIEW_INLINE_SIZE)
                           ? reinterpret_cast<char const*>(item.inlined.data)
                           : d_ptrs[item.ref.buffer_index] + item.ref.offset;
      return {data, size};
    });

  return cudf::strings::detail::make_strings_column(d_indices.begin(), d_indices.end(), stream, mr);
}

}  // namespace

std::unique_ptr<column> string_column_from_arrow_host(ArrowSchemaView* schema,
                                                      ArrowArray const* input,
                                                      std::unique_ptr<rmm::device_buffer>&& mask,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  return schema->type == NANOARROW_TYPE_STRING_VIEW
           ? from_arrow_stringview(schema, input, std::move(mask), stream, mr)
           : from_arrow_string(schema, input, std::move(mask), stream, mr);
}

}  // namespace detail
}  // namespace cudf
