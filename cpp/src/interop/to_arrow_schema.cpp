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

#include "to_arrow_utilities.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/interop/detail/arrow.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>

namespace cudf {
namespace detail {
namespace {

struct dispatch_to_arrow_type {
  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>())>
  int operator()(column_view, column_metadata const&, ArrowSchema*)
  {
    CUDF_FAIL("Unsupported type for to_arrow_schema", cudf::data_type_error);
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  int operator()(column_view input_view, column_metadata const&, ArrowSchema* out)
  {
    cudf::type_id id = input_view.type().id();
    switch (id) {
      case cudf::type_id::TIMESTAMP_SECONDS:
        return ArrowSchemaSetTypeDateTime(
          out, NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_SECOND, nullptr);
      case cudf::type_id::TIMESTAMP_MILLISECONDS:
        return ArrowSchemaSetTypeDateTime(
          out, NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MILLI, nullptr);
      case cudf::type_id::TIMESTAMP_MICROSECONDS:
        return ArrowSchemaSetTypeDateTime(
          out, NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MICRO, nullptr);
      case cudf::type_id::TIMESTAMP_NANOSECONDS:
        return ArrowSchemaSetTypeDateTime(
          out, NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_NANO, nullptr);
      case cudf::type_id::DURATION_SECONDS:
        return ArrowSchemaSetTypeDateTime(
          out, NANOARROW_TYPE_DURATION, NANOARROW_TIME_UNIT_SECOND, nullptr);
      case cudf::type_id::DURATION_MILLISECONDS:
        return ArrowSchemaSetTypeDateTime(
          out, NANOARROW_TYPE_DURATION, NANOARROW_TIME_UNIT_MILLI, nullptr);
      case cudf::type_id::DURATION_MICROSECONDS:
        return ArrowSchemaSetTypeDateTime(
          out, NANOARROW_TYPE_DURATION, NANOARROW_TIME_UNIT_MICRO, nullptr);
      case cudf::type_id::DURATION_NANOSECONDS:
        return ArrowSchemaSetTypeDateTime(
          out, NANOARROW_TYPE_DURATION, NANOARROW_TIME_UNIT_NANO, nullptr);
      default: return ArrowSchemaSetType(out, id_to_arrow_type(id));
    }
  }
};

template <typename DeviceType>
int decimals_to_arrow(column_view input, ArrowSchema* out)
{
  // Arrow doesn't support decimal32/decimal64 currently. decimal128
  // is the smallest that arrow supports besides float32/float64 so we
  // upcast to decimal128.
  return ArrowSchemaSetTypeDecimal(out,
                                   NANOARROW_TYPE_DECIMAL128,
                                   cudf::detail::max_precision<DeviceType>(),
                                   -input.type().scale());
}

template <>
int dispatch_to_arrow_type::operator()<numeric::decimal32>(column_view input,
                                                           column_metadata const&,
                                                           ArrowSchema* out)
{
  using DeviceType = int32_t;
  return decimals_to_arrow<DeviceType>(input, out);
}

template <>
int dispatch_to_arrow_type::operator()<numeric::decimal64>(column_view input,
                                                           column_metadata const&,
                                                           ArrowSchema* out)
{
  using DeviceType = int64_t;
  return decimals_to_arrow<DeviceType>(input, out);
}

template <>
int dispatch_to_arrow_type::operator()<numeric::decimal128>(column_view input,
                                                            column_metadata const&,
                                                            ArrowSchema* out)
{
  using DeviceType = __int128_t;
  return decimals_to_arrow<DeviceType>(input, out);
}

template <>
int dispatch_to_arrow_type::operator()<cudf::string_view>(column_view input,
                                                          column_metadata const&,
                                                          ArrowSchema* out)
{
  return ArrowSchemaSetType(out, NANOARROW_TYPE_STRING);
}

// these forward declarations are needed due to the recursive calls to them
// inside their definitions and in struct_vew for handling children
template <>
int dispatch_to_arrow_type::operator()<cudf::list_view>(column_view input,
                                                        column_metadata const& metadata,
                                                        ArrowSchema* out);

template <>
int dispatch_to_arrow_type::operator()<cudf::dictionary32>(column_view input,
                                                           column_metadata const& metadata,
                                                           ArrowSchema* out);

template <>
int dispatch_to_arrow_type::operator()<cudf::struct_view>(column_view input,
                                                          column_metadata const& metadata,
                                                          ArrowSchema* out)
{
  CUDF_EXPECTS(metadata.children_meta.size() == static_cast<std::size_t>(input.num_children()),
               "Number of field names and number of children doesn't match\n");

  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(out, input.num_children()));
  for (int i = 0; i < input.num_children(); ++i) {
    auto child = out->children[i];
    auto col   = input.child(i);
    ArrowSchemaInit(child);
    NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(child, metadata.children_meta[i].name.c_str()));

    child->flags = col.has_nulls() ? ARROW_FLAG_NULLABLE : 0;

    NANOARROW_RETURN_NOT_OK(cudf::type_dispatcher(
      col.type(), detail::dispatch_to_arrow_type{}, col, metadata.children_meta[i], child));
  }

  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_type::operator()<cudf::list_view>(column_view input,
                                                        column_metadata const& metadata,
                                                        ArrowSchema* out)
{
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(out, NANOARROW_TYPE_LIST));
  auto child = input.child(cudf::lists_column_view::child_column_index);
  ArrowSchemaInit(out->children[0]);
  auto child_meta =
    metadata.children_meta.empty() ? column_metadata{"element"} : metadata.children_meta[0];

  out->flags = input.has_nulls() ? ARROW_FLAG_NULLABLE : 0;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(out->children[0], child_meta.name.c_str()));
  out->children[0]->flags = child.has_nulls() ? ARROW_FLAG_NULLABLE : 0;
  return cudf::type_dispatcher(
    child.type(), detail::dispatch_to_arrow_type{}, child, child_meta, out->children[0]);
}

template <>
int dispatch_to_arrow_type::operator()<cudf::dictionary32>(column_view input,
                                                           column_metadata const& metadata,
                                                           ArrowSchema* out)
{
  cudf::dictionary_column_view dview{input};

  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(out, id_to_arrow_type(dview.indices().type().id())));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateDictionary(out));
  ArrowSchemaInit(out->dictionary);

  auto dict_keys = dview.keys();
  return cudf::type_dispatcher(
    dict_keys.type(),
    detail::dispatch_to_arrow_type{},
    dict_keys,
    metadata.children_meta.empty() ? column_metadata{"keys"} : metadata.children_meta[0],
    out->dictionary);
}
}  // namespace
}  // namespace detail

unique_schema_t to_arrow_schema(cudf::table_view const& input,
                                cudf::host_span<column_metadata const> metadata)
{
  CUDF_EXPECTS((metadata.size() == static_cast<std::size_t>(input.num_columns())),
               "columns' metadata should be equal to the number of columns in table");

  nanoarrow::UniqueSchema result;
  ArrowSchemaInit(result.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(result.get(), input.num_columns()));

  for (int i = 0; i < input.num_columns(); ++i) {
    auto child = result->children[i];
    auto col   = input.column(i);
    ArrowSchemaInit(child);
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child, metadata[i].name.c_str()));
    child->flags = col.has_nulls() ? ARROW_FLAG_NULLABLE : 0;

    NANOARROW_THROW_NOT_OK(
      cudf::type_dispatcher(col.type(), detail::dispatch_to_arrow_type{}, col, metadata[i], child));
  }

  unique_schema_t out(new ArrowSchema, [](ArrowSchema* schema) {
    if (schema->release != nullptr) { ArrowSchemaRelease(schema); }
    delete schema;
  });
  result.move(out.get());
  return out;
}

}  // namespace cudf
