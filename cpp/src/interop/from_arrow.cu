/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

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
      arrow::TimestampType const* type = static_cast<arrow::TimestampType const*>(&arrow_type);
      switch (type->unit()) {
        case arrow::TimeUnit::type::SECOND: return data_type(type_id::TIMESTAMP_SECONDS);
        case arrow::TimeUnit::type::MILLI: return data_type(type_id::TIMESTAMP_MILLISECONDS);
        case arrow::TimeUnit::type::MICRO: return data_type(type_id::TIMESTAMP_MICROSECONDS);
        case arrow::TimeUnit::type::NANO: return data_type(type_id::TIMESTAMP_NANOSECONDS);
        default: CUDF_FAIL("Unsupported timestamp unit in arrow");
      }
    }
    case arrow::Type::STRING: return data_type(type_id::STRING);
    case arrow::Type::DICTIONARY: return data_type(type_id::DICTIONARY32);
    case arrow::Type::LIST: return data_type(type_id::LIST);
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
   * @brief Returns mask from an array withut any offsets.
   */
  std::unique_ptr<rmm::device_buffer> get_mask_buffer(arrow::Array const& array,
                                                      rmm::mr::device_memory_resource* mr,
                                                      cudaStream_t stream)
  {
    if (array.null_bitmap_data() == nullptr) {
      return std::make_unique<rmm::device_buffer>(0, stream, mr);
    }
    auto mask = std::make_unique<rmm::device_buffer>(
      bitmask_allocation_size_bytes(static_cast<size_type>(array.null_bitmap()->size() * CHAR_BIT)),
      stream,
      mr);
    CUDA_TRY(cudaMemcpyAsync(mask->data(),
                             array.null_bitmap_data(),
                             array.null_bitmap()->size(),
                             cudaMemcpyHostToDevice,
                             stream));
    return mask;
  }

  template <typename T>
  std::unique_ptr<column> operator()(arrow::Array const& array,
                                     data_type type,
                                     bool skip_mask,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream)
  {
    auto data_buffer         = array.data()->buffers[1];
    size_type const num_rows = array.length();
    auto const has_nulls     = skip_mask ? false : array.null_bitmap_data() != nullptr;
    auto col = make_fixed_width_column(type, num_rows, mask_state::UNALLOCATED, stream, mr);
    auto mutable_column_view = col->mutable_view();
    CUDA_TRY(cudaMemcpyAsync(mutable_column_view.data<void*>(),
                             data_buffer->data() + array.offset() * sizeof(T),
                             sizeof(T) * num_rows,
                             cudaMemcpyHostToDevice,
                             stream));
    if (has_nulls) {
      auto tmp_mask = get_mask_buffer(array, mr, stream);

      // If array is sliced, we have to copy whole mask and then take copy.
      auto out_mask = (num_rows == static_cast<size_type>(data_buffer->size() / sizeof(T)))
                        ? *tmp_mask
                        : copy_bitmask(static_cast<bitmask_type*>(tmp_mask->data()),
                                       array.offset(),
                                       array.offset() + num_rows,
                                       stream,
                                       mr);

      col->set_null_mask(std::move(out_mask));
    }

    return std::move(col);
  }
};

/**
 * @brief Returns cudf column formed from given arrow array
 * This has been introduced to take care of compiler error "error: explicit specialization of
 * function must precede its first use"
 */
std::unique_ptr<column> get_column(arrow::Array const& array,
                                   data_type type,
                                   bool skip_mask,
                                   rmm::mr::device_memory_resource* mr,
                                   cudaStream_t stream);

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<bool>(
  arrow::Array const& array,
  data_type type,
  bool skip_mask,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  auto data_buffer = array.data()->buffers[1];
  auto data        = rmm::device_buffer(data_buffer->size(), stream, mr);
  CUDA_TRY(cudaMemcpyAsync(
    data.data(), data_buffer->data(), data_buffer->size(), cudaMemcpyHostToDevice, stream));
  auto out_col = mask_to_bools(static_cast<bitmask_type*>(data.data()),
                               array.offset(),
                               array.offset() + array.length(),
                               stream,
                               mr);

  auto const has_nulls = skip_mask ? false : array.null_bitmap_data() != nullptr;
  if (has_nulls) {
    auto out_mask =
      copy_bitmask(static_cast<bitmask_type*>(get_mask_buffer(array, mr, stream)->data()),
                   array.offset(),
                   array.offset() + array.length(),
                   stream,
                   mr);

    out_col->set_null_mask(std::move(out_mask));
  }

  return out_col;
}

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<cudf::string_view>(
  arrow::Array const& array,
  data_type type,
  bool skip_mask,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  if (array.length() == 0) { return cudf::strings::detail::make_empty_strings_column(mr, stream); }
  auto str_array    = static_cast<arrow::StringArray const*>(&array);
  auto offset_array = std::make_unique<arrow::Int32Array>(
    str_array->value_offsets()->size() / sizeof(int32_t), str_array->value_offsets(), nullptr);
  auto char_array = std::make_unique<arrow::Int8Array>(
    str_array->value_data()->size(), str_array->value_data(), nullptr);

  auto offsets_column = dispatch_to_cudf_column{}.operator()<int32_t>(
    *offset_array, data_type(type_id::INT32), true, mr, stream);
  auto chars_column = dispatch_to_cudf_column{}.operator()<int8_t>(
    *char_array, data_type(type_id::INT8), true, mr, stream);

  auto const num_rows = offsets_column->size() - 1;
  auto out_col        = make_strings_column(num_rows,
                                     std::move(offsets_column),
                                     std::move(chars_column),
                                     UNKNOWN_NULL_COUNT,
                                     std::move(*get_mask_buffer(array, mr, stream)),
                                     stream,
                                     mr);

  return num_rows == array.length() ? std::move(out_col)
                                    : std::make_unique<column>(cudf::detail::slice(
                                        out_col->view(),
                                        static_cast<size_type>(array.offset()),
                                        static_cast<size_type>(array.offset() + array.length())));
}

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<cudf::dictionary32>(
  arrow::Array const& array,
  data_type type,
  bool skip_mask,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  auto dict_array     = static_cast<arrow::DictionaryArray const*>(&array);
  auto ind_type       = arrow_to_cudf_type(*(dict_array->indices()->type()));
  auto indices_column = get_column(*(dict_array->indices()), ind_type, false, mr, stream);
  // If index type is not of type int32_t, then cast it to int32_t
  if (indices_column->type().id() != type_id::INT32)
    indices_column =
      cudf::detail::cast(indices_column->view(), data_type(type_id::INT32), mr, stream);

  auto dict_type   = arrow_to_cudf_type(*(dict_array->dictionary()->type()));
  auto keys_column = get_column(*(dict_array->dictionary()), dict_type, true, mr, stream);

  // Child columns shouldn't have masks and we need the mask in main column
  auto column_contents = indices_column->release();
  indices_column       = std::make_unique<column>(data_type(type_id::INT32),
                                            static_cast<size_type>(array.length()),
                                            std::move(*(column_contents.data)));

  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(column_contents.null_mask)),
                                UNKNOWN_NULL_COUNT);
}

template <>
std::unique_ptr<column> dispatch_to_cudf_column::operator()<cudf::list_view>(
  arrow::Array const& array,
  data_type type,
  bool skip_mask,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream)
{
  auto list_array   = static_cast<arrow::ListArray const*>(&array);
  auto offset_array = std::make_unique<arrow::Int32Array>(
    list_array->value_offsets()->size() / sizeof(int32_t), list_array->value_offsets(), nullptr);
  auto offsets_column = dispatch_to_cudf_column{}.operator()<int32_t>(
    *offset_array, data_type(type_id::INT32), true, mr, stream);

  auto child_type   = arrow_to_cudf_type(*(list_array->values()->type()));
  auto child_column = get_column(*(list_array->values()), child_type, false, mr, stream);

  auto const num_rows = offsets_column->size() - 1;
  auto out_col        = make_lists_column(num_rows,
                                   std::move(offsets_column),
                                   std::move(child_column),
                                   UNKNOWN_NULL_COUNT,
                                   std::move(*get_mask_buffer(array, mr, stream)));

  return num_rows == array.length() ? std::move(out_col)
                                    : std::make_unique<column>(cudf::detail::slice(
                                        out_col->view(),
                                        static_cast<size_type>(array.offset()),
                                        static_cast<size_type>(array.offset() + array.length())));
}

std::unique_ptr<column> get_column(arrow::Array const& array,
                                   data_type type,
                                   bool skip_mask,
                                   rmm::mr::device_memory_resource* mr,
                                   cudaStream_t stream)
{
  return type_dispatcher(type, dispatch_to_cudf_column{}, array, type, skip_mask, mr, stream);
}

}  // namespace

std::unique_ptr<table> from_arrow(arrow::Table const& input_table,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
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
                     return std::make_unique<column>(
                       cudf_type, chunked_array->length(), std::move(rmm::device_buffer(0)));
                   }
                   transform(array_chunks.begin(),
                             array_chunks.end(),
                             std::back_inserter(concat_columns),
                             [&cudf_type, &mr, &stream](auto const& array_chunk) {
                               return get_column(*array_chunk, cudf_type, false, mr, stream);
                             });
                   if (concat_columns.size() == 0) {
                     return std::make_unique<column>(cudf_type, 0, rmm::device_buffer(0));
                   } else if (concat_columns.size() == 1) {
                     return std::move(concat_columns[0]);
                   }

                   std::vector<cudf::column_view> column_views;
                   std::transform(concat_columns.begin(),
                                  concat_columns.end(),
                                  std::back_inserter(column_views),
                                  [](auto const& col) { return col->view(); });
                   return cudf::detail::concatenate(column_views, mr, stream);
                 });

  return std::make_unique<table>(std::move(columns));
}

}  // namespace detail

std::unique_ptr<table> from_arrow(arrow::Table const& input_table,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow(input_table, mr);
}

}  // namespace cudf
