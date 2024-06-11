/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/gather.h>

namespace cudf {

namespace detail {
data_type arrow_to_cudf_type(arrow::DataType const& arrow_type)
{
  switch (arrow_type.id()) {
    case arrow::Type::NA: return data_type(type_id::EMPTY);
    case arrow::Type::BOOL: return data_type(type_id::BOOL8);
    case arrow::Type::INT8: return data_type(type_id::INT8);
    case arrow::Type::INT16: return data_type(type_id::INT16);
    case arrow::Type::INT32: return data_type(type_id::INT32);
    case arrow::Type::INT64: return data_type(type_id::INT64);
    case arrow::Type::UINT8: return data_type(type_id::UINT8);
    case arrow::Type::UINT16: return data_type(type_id::UINT16);
    case arrow::Type::UINT32: return data_type(type_id::UINT32);
    case arrow::Type::UINT64: return data_type(type_id::UINT64);
    case arrow::Type::FLOAT: return data_type(type_id::FLOAT32);
    case arrow::Type::DOUBLE: return data_type(type_id::FLOAT64);
    case arrow::Type::DATE32: return data_type(type_id::TIMESTAMP_DAYS);
    case arrow::Type::TIMESTAMP: {
      auto type = static_cast<arrow::TimestampType const*>(&arrow_type);
      switch (type->unit()) {
        case arrow::TimeUnit::type::SECOND: return data_type(type_id::TIMESTAMP_SECONDS);
        case arrow::TimeUnit::type::MILLI: return data_type(type_id::TIMESTAMP_MILLISECONDS);
        case arrow::TimeUnit::type::MICRO: return data_type(type_id::TIMESTAMP_MICROSECONDS);
        case arrow::TimeUnit::type::NANO: return data_type(type_id::TIMESTAMP_NANOSECONDS);
        default: CUDF_FAIL("Unsupported timestamp unit in arrow");
      }
    }
    case arrow::Type::DURATION: {
      auto type = static_cast<arrow::DurationType const*>(&arrow_type);
      switch (type->unit()) {
        case arrow::TimeUnit::type::SECOND: return data_type(type_id::DURATION_SECONDS);
        case arrow::TimeUnit::type::MILLI: return data_type(type_id::DURATION_MILLISECONDS);
        case arrow::TimeUnit::type::MICRO: return data_type(type_id::DURATION_MICROSECONDS);
        case arrow::TimeUnit::type::NANO: return data_type(type_id::DURATION_NANOSECONDS);
        default: CUDF_FAIL("Unsupported duration unit in arrow");
      }
    }
    case arrow::Type::STRING: return data_type(type_id::STRING);
    case arrow::Type::LARGE_STRING: return data_type(type_id::STRING);
    case arrow::Type::DICTIONARY: return data_type(type_id::DICTIONARY32);
    case arrow::Type::LIST: return data_type(type_id::LIST);
    case arrow::Type::DECIMAL: {
      auto const type = static_cast<arrow::Decimal128Type const*>(&arrow_type);
      return data_type{type_id::DECIMAL128, -type->scale()};
    }
    case arrow::Type::STRUCT: return data_type(type_id::STRUCT);
    default: CUDF_FAIL("Unsupported type_id conversion to cudf");
  }
}

namespace {
/**
 * @brief Functor to return column for a corresponding arrow array. column
 * is formed from buffer underneath the arrow array along with any offset and
 * change in length that array has.
 */
struct dispatch_to_cudf_column {
  /**
   * @brief Returns mask from an array without any offsets.
   */
  std::unique_ptr<rmm::device_buffer> get_mask_buffer(arrow::Array const& array,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
  {
    if (array.null_bitmap_data() == nullptr) {
      return std::make_unique<rmm::device_buffer>(0, stream, mr);
    }
    auto const null_bitmap_size = array.null_bitmap()->size();
    auto const allocation_size =
      bitmask_allocation_size_bytes(static_cast<size_type>(null_bitmap_size * CHAR_BIT));
    auto mask        = std::make_unique<rmm::device_buffer>(allocation_size, stream, mr);
    auto mask_buffer = array.null_bitmap();
    CUDF_CUDA_TRY(cudaMemcpyAsync(mask->data(),
                                  reinterpret_cast<uint8_t const*>(mask_buffer->address()),
                                  null_bitmap_size,
                                  cudaMemcpyDefault,
                                  stream.value()));
    // Zero-initialize trailing padding bytes
    auto const num_trailing_bytes = allocation_size - null_bitmap_size;
    if (num_trailing_bytes > 0) {
      auto trailing_bytes = static_cast<uint8_t*>(mask->data()) + null_bitmap_size;
      CUDF_CUDA_TRY(cudaMemsetAsync(trailing_bytes, 0, num_trailing_bytes, stream.value()));
    }
    return mask;
  }

  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>())>
  std::unique_ptr<column> operator()(
    arrow::Array const&, data_type, bool, rmm::cuda_stream_view, rmm::device_async_resource_ref)
  {
    CUDF_FAIL("Unsupported type in from_arrow.");
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  std::unique_ptr<column> operator()(arrow::Array const& array,
                                     data_type type,
                                     bool skip_mask,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    auto data_buffer         = array.data()->buffers[1];
    size_type const num_rows = array.length();
    auto const has_nulls     = skip_mask ? false : array.null_bitmap_data() != nullptr;
    auto col = make_fixed_width_column(type, num_rows, mask_state::UNALLOCATED, stream, mr);
    auto mutable_column_view = col->mutable_view();
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      mutable_column_view.data<T>(),
      reinterpret_cast<uint8_t const*>(data_buffer->address()) + array.offset() * sizeof(T),
      sizeof(T) * num_rows,
      cudaMemcpyDefault,
      stream.value()));
    if (has_nulls) {
      auto tmp_mask = get_mask_buffer(array, stream, mr);

      // If array is sliced, we have to copy whole mask and then take copy.
      auto out_mask = (num_rows == static_cast<size_type>(data_buffer->size() / sizeof(T)))
                        ? std::move(*tmp_mask)
                        : cudf::detail::copy_bitmask(static_cast<bitmask_type*>(tmp_mask->data()),
                                                     array.offset(),
                                                     array.offset() + num_rows,
                                                     stream,
                                                     mr);

      col->set_null_mask(std::move(out_mask), array.null_count());
    }

    return col;
  }
};

std::unique_ptr<column> get_empty_type_column(size_type size)
{
  // this abomination is required by cuDF Python, which needs to handle
  // [PyArrow null arrays](https://arrow.apache.org/docs/python/generated/pyarrow.NullArray.html)
  // of finite length
  return std::make_unique<column>(
    data_type(type_id::EMPTY), size, rmm::device_buffer{}, rmm::device_buffer{}, size);
}

/**
 * @brief Returns cudf column formed from given arrow array
 * This has been introduced to take care of compiler error "error: explicit specialization of
 * function must precede its first use"
 */
std::unique_ptr<column> get_column(arrow::Array const& array,
                                   data_type type,
                                   bool skip_mask,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<numeric::decimal128>(
  arrow::Array const& array,
  data_type type,
  bool skip_mask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  using DeviceType = __int128_t;

  auto data_buffer    = array.data()->buffers[1];
  auto const num_rows = static_cast<size_type>(array.length());
  auto col = make_fixed_width_column(type, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto mutable_column_view = col->mutable_view();

  CUDF_CUDA_TRY(cudaMemcpyAsync(
    mutable_column_view.data<DeviceType>(),
    reinterpret_cast<uint8_t const*>(data_buffer->address()) + array.offset() * sizeof(DeviceType),
    sizeof(DeviceType) * num_rows,
    cudaMemcpyDefault,
    stream.value()));

  auto null_mask = [&] {
    if (not skip_mask and array.null_bitmap_data()) {
      auto temp_mask = get_mask_buffer(array, stream, mr);
      // If array is sliced, we have to copy whole mask and then take copy.
      return (num_rows == static_cast<size_type>(data_buffer->size() / sizeof(DeviceType)))
               ? std::move(*temp_mask.release())
               : cudf::detail::copy_bitmask(static_cast<bitmask_type*>(temp_mask->data()),
                                            array.offset(),
                                            array.offset() + num_rows,
                                            stream,
                                            mr);
    }
    return rmm::device_buffer{};
  }();

  col->set_null_mask(std::move(null_mask), array.null_count());
  return col;
}

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<bool>(arrow::Array const& array,
                                                                  data_type,
                                                                  bool skip_mask,
                                                                  rmm::cuda_stream_view stream,
                                                                  rmm::device_async_resource_ref mr)
{
  auto data_buffer = array.data()->buffers[1];
  // mask-to-bools expects the mask to be bitmask_type aligned/padded
  auto data = rmm::device_buffer(
    cudf::bitmask_allocation_size_bytes(data_buffer->size() * CHAR_BIT), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(data.data(),
                                reinterpret_cast<uint8_t const*>(data_buffer->address()),
                                data_buffer->size(),
                                cudaMemcpyDefault,
                                stream.value()));
  auto out_col = mask_to_bools(static_cast<bitmask_type*>(data.data()),
                               array.offset(),
                               array.offset() + array.length(),
                               stream,
                               mr);

  auto const has_nulls = skip_mask ? false : array.null_bitmap_data() != nullptr;
  if (has_nulls) {
    auto out_mask =
      detail::copy_bitmask(static_cast<bitmask_type*>(get_mask_buffer(array, stream, mr)->data()),
                           array.offset(),
                           array.offset() + array.length(),
                           stream,
                           mr);

    out_col->set_null_mask(std::move(out_mask), array.null_count());
  }

  return out_col;
}

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<cudf::string_view>(
  arrow::Array const& array,
  data_type,
  bool,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (array.length() == 0) { return make_empty_column(type_id::STRING); }

  std::unique_ptr<column> offsets_column;
  std::unique_ptr<arrow::Array> char_array;

  if (array.type_id() == arrow::Type::LARGE_STRING) {
    auto str_array    = static_cast<arrow::LargeStringArray const*>(&array);
    auto offset_array = std::make_unique<arrow::Int64Array>(
      str_array->value_offsets()->size() / sizeof(int64_t), str_array->value_offsets(), nullptr);
    offsets_column = dispatch_to_cudf_column{}.operator()<int64_t>(
      *offset_array, data_type(type_id::INT64), true, stream, mr);
    char_array = std::make_unique<arrow::Int8Array>(
      str_array->value_data()->size(), str_array->value_data(), nullptr);
  } else if (array.type_id() == arrow::Type::STRING) {
    auto str_array    = static_cast<arrow::StringArray const*>(&array);
    auto offset_array = std::make_unique<arrow::Int32Array>(
      str_array->value_offsets()->size() / sizeof(int32_t), str_array->value_offsets(), nullptr);
    offsets_column = dispatch_to_cudf_column{}.operator()<int32_t>(
      *offset_array, data_type(type_id::INT32), true, stream, mr);
    char_array = std::make_unique<arrow::Int8Array>(
      str_array->value_data()->size(), str_array->value_data(), nullptr);
  } else {
    throw std::runtime_error("Unsupported array type");
  }

  rmm::device_buffer chars(char_array->length(), stream, mr);
  auto data_buffer = char_array->data()->buffers[1];
  CUDF_CUDA_TRY(cudaMemcpyAsync(chars.data(),
                                reinterpret_cast<uint8_t const*>(data_buffer->address()),
                                chars.size(),
                                cudaMemcpyDefault,
                                stream.value()));

  auto const num_rows = offsets_column->size() - 1;
  auto out_col        = make_strings_column(num_rows,
                                     std::move(offsets_column),
                                     std::move(chars),
                                     array.null_count(),
                                     std::move(*get_mask_buffer(array, stream, mr)));

  return num_rows == array.length()
           ? std::move(out_col)
           : std::make_unique<column>(
               cudf::detail::slice(out_col->view(),
                                   static_cast<size_type>(array.offset()),
                                   static_cast<size_type>(array.offset() + array.length()),
                                   stream),
               stream,
               mr);
}

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<cudf::dictionary32>(
  arrow::Array const& array,
  data_type,
  bool,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto dict_array  = static_cast<arrow::DictionaryArray const*>(&array);
  auto dict_type   = arrow_to_cudf_type(*(dict_array->dictionary()->type()));
  auto keys_column = get_column(*(dict_array->dictionary()), dict_type, true, stream, mr);
  auto ind_type    = arrow_to_cudf_type(*(dict_array->indices()->type()));

  auto indices_column = get_column(*(dict_array->indices()), ind_type, false, stream, mr);
  // If index type is not of type uint32_t, then cast it to uint32_t
  auto const dict_indices_type = data_type{type_id::UINT32};
  if (indices_column->type().id() != dict_indices_type.id())
    indices_column = cudf::detail::cast(indices_column->view(), dict_indices_type, stream, mr);

  // Child columns shouldn't have masks and we need the mask in main column
  auto column_contents = indices_column->release();
  indices_column       = std::make_unique<column>(dict_indices_type,
                                            static_cast<size_type>(array.length()),
                                            std::move(*(column_contents.data)),
                                            rmm::device_buffer{},
                                            0);

  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(column_contents.null_mask)),
                                array.null_count());
}

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<cudf::struct_view>(
  arrow::Array const& array,
  data_type,
  bool,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto struct_array = static_cast<arrow::StructArray const*>(&array);
  std::vector<std::unique_ptr<column>> child_columns;
  // Offsets have already been applied to child
  arrow::ArrayVector array_children = struct_array->fields();
  std::transform(array_children.cbegin(),
                 array_children.cend(),
                 std::back_inserter(child_columns),
                 [&mr, &stream](auto const& child_array) {
                   auto type = arrow_to_cudf_type(*(child_array->type()));
                   return get_column(*child_array, type, false, stream, mr);
                 });

  auto out_mask = std::move(*(get_mask_buffer(array, stream, mr)));
  if (struct_array->null_bitmap_data() != nullptr) {
    out_mask = detail::copy_bitmask(static_cast<bitmask_type*>(out_mask.data()),
                                    array.offset(),
                                    array.offset() + array.length(),
                                    stream,
                                    mr);
  }

  return make_structs_column(
    array.length(), move(child_columns), array.null_count(), std::move(out_mask), stream, mr);
}

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<cudf::list_view>(
  arrow::Array const& array,
  data_type,
  bool,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto list_array   = static_cast<arrow::ListArray const*>(&array);
  auto offset_array = std::make_unique<arrow::Int32Array>(
    list_array->value_offsets()->size() / sizeof(int32_t), list_array->value_offsets(), nullptr);
  auto offsets_column = dispatch_to_cudf_column{}.operator()<int32_t>(
    *offset_array, data_type(type_id::INT32), true, stream, mr);

  auto child_type   = arrow_to_cudf_type(*(list_array->values()->type()));
  auto child_column = get_column(*(list_array->values()), child_type, false, stream, mr);

  auto const num_rows = offsets_column->size() - 1;
  auto out_col        = make_lists_column(num_rows,
                                   std::move(offsets_column),
                                   std::move(child_column),
                                   array.null_count(),
                                   std::move(*get_mask_buffer(array, stream, mr)),
                                   stream,
                                   mr);

  return num_rows == array.length()
           ? std::move(out_col)
           : std::make_unique<column>(
               cudf::detail::slice(out_col->view(),
                                   static_cast<size_type>(array.offset()),
                                   static_cast<size_type>(array.offset() + array.length()),
                                   stream),
               stream,
               mr);
}

std::unique_ptr<column> get_column(arrow::Array const& array,
                                   data_type type,
                                   bool skip_mask,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  return type.id() != type_id::EMPTY
           ? type_dispatcher(type, dispatch_to_cudf_column{}, array, type, skip_mask, stream, mr)
           : get_empty_type_column(array.length());
}

}  // namespace

std::unique_ptr<table> from_arrow(arrow::Table const& input_table,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  if (input_table.num_columns() == 0) { return std::make_unique<table>(); }
  std::vector<std::unique_ptr<column>> columns;
  auto chunked_arrays = input_table.columns();
  std::transform(chunked_arrays.begin(),
                 chunked_arrays.end(),
                 std::back_inserter(columns),
                 [&mr, &stream](auto const& chunked_array) {
                   std::vector<std::unique_ptr<column>> concat_columns;
                   auto cudf_type    = arrow_to_cudf_type(*(chunked_array->type()));
                   auto array_chunks = chunked_array->chunks();
                   if (cudf_type.id() == type_id::EMPTY) {
                     return get_empty_type_column(chunked_array->length());
                   }
                   std::transform(array_chunks.begin(),
                                  array_chunks.end(),
                                  std::back_inserter(concat_columns),
                                  [&cudf_type, &mr, &stream](auto const& array_chunk) {
                                    return get_column(*array_chunk, cudf_type, false, stream, mr);
                                  });
                   if (concat_columns.empty()) {
                     return std::make_unique<column>(
                       cudf_type, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
                   } else if (concat_columns.size() == 1) {
                     return std::move(concat_columns[0]);
                   }

                   std::vector<cudf::column_view> column_views;
                   std::transform(concat_columns.begin(),
                                  concat_columns.end(),
                                  std::back_inserter(column_views),
                                  [](auto const& col) { return col->view(); });
                   return cudf::detail::concatenate(column_views, stream, mr);
                 });

  return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<cudf::scalar> from_arrow(arrow::Scalar const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto maybe_array = arrow::MakeArrayFromScalar(input, 1);
  if (!maybe_array.ok()) { CUDF_FAIL("Failed to create array"); }
  auto array = *maybe_array;

  auto field = arrow::field("", input.type);

  auto table = arrow::Table::Make(arrow::schema({field}), {array});

  auto cudf_table = detail::from_arrow(*table, stream, mr);

  auto cv = cudf_table->view().column(0);
  return get_element(cv, 0, stream);
}

}  // namespace detail

std::unique_ptr<table> from_arrow(arrow::Table const& input_table,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow(input_table, stream, mr);
}

std::unique_ptr<cudf::scalar> from_arrow(arrow::Scalar const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow(input, stream, mr);
}
}  // namespace cudf
