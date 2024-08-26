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

#include "cudf/column/column.hpp"
#include "cudf/column/column_view.hpp"
#include "cudf/io/json.hpp"
#include "cudf/io/types.hpp"
#include "cudf/lists/extract.hpp"
#include "cudf/scalar/scalar.hpp"
#include "rmm/device_buffer.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"
#include "thrust/fill.h"
#include "thrust/find.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/io/new_json_object.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/unary.hpp>

#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

// #include "cub/device/device_memcpy.cuh"
#include "cub/device/device_histogram.cuh"

namespace cudf {
namespace io {

void print_schema(cudf::io::schema_element const& sch)
{
  // std::cout << (sch.type.id()==type_id::LIST ? "LIST." : sch.type.id()==type_id::STRUCT ?
  // "STRUCT." : sch.type.id()==type_id::STRING ? "STRING." : "OTHER.");
  std::cout << (sch.type.id() == cudf::type_id::STRING ? "STRING." : "");
  for (auto ch : sch.child_types) {
    std::cout << ch.first << ">";
    print_schema(ch.second);
  }
}

// Convert the token value into string name, for debugging purpose.
std::string token_to_string(cudf::io::json::PdaTokenT const token_type)
{
  switch (token_type) {
    case cudf::io::json::token_t::StructBegin: return "StructBegin";
    case cudf::io::json::token_t::StructEnd: return "StructEnd";
    case cudf::io::json::token_t::ListBegin: return "ListBegin";
    case cudf::io::json::token_t::ListEnd: return "ListEnd";
    case cudf::io::json::token_t::StructMemberBegin: return "StructMemberBegin";
    case cudf::io::json::token_t::StructMemberEnd: return "StructMemberEnd";
    case cudf::io::json::token_t::FieldNameBegin: return "FieldNameBegin";
    case cudf::io::json::token_t::FieldNameEnd: return "FieldNameEnd";
    case cudf::io::json::token_t::StringBegin: return "StringBegin";
    case cudf::io::json::token_t::StringEnd: return "StringEnd";
    case cudf::io::json::token_t::ValueBegin: return "ValueBegin";
    case cudf::io::json::token_t::ValueEnd: return "ValueEnd";
    case cudf::io::json::token_t::ErrorBegin: return "ErrorBegin";
    case cudf::io::json::token_t::LineEnd: return "LineEnd";
    default: return "Unknown";
  }
}

// Print the content of the input device vector.
template <typename T, typename U = int>
void print_debug(rmm::device_uvector<T> const& input,
                 std::string const& name,
                 std::string const& separator,
                 rmm::cuda_stream_view stream)
{
  auto const h_input = cudf::detail::make_host_vector_sync(
    cudf::device_span<T const>{input.data(), input.size()}, stream);
  std::stringstream ss;
  ss << name << ":\n";
  for (size_t i = 0; i < h_input.size(); ++i) {
    ss << static_cast<U>(h_input[i]);
    if (separator.size() > 0 && i + 1 < h_input.size()) { ss << separator; }
  }
  std::cerr << ss.str() << std::endl;
}

// Print the content of the input device vector.
void print_debug_tokens(rmm::device_uvector<cudf::io::json::PdaTokenT> const& tokens,
                        rmm::device_uvector<uint32_t> const& offsets,
                        rmm::device_uvector<char> const& str_data,
                        std::string const& name,
                        std::string const& separator,
                        rmm::cuda_stream_view stream)
{
  auto const h_tokens = cudf::detail::make_host_vector_sync(
    cudf::device_span<cudf::io::json::PdaTokenT const>{tokens.data(), tokens.size()}, stream);
  auto const h_offsets = cudf::detail::make_host_vector_sync(
    cudf::device_span<uint32_t const>{offsets.data(), offsets.size()}, stream);
  auto const h_str_data = cudf::detail::make_host_vector_sync(
    cudf::device_span<char const>{str_data.data(), str_data.size()}, stream);

  std::stringstream ss;
  ss << name << ":\n";
  uint32_t str_begin = 0;
  for (size_t i = 0; i < h_tokens.size(); ++i) {
    ss << token_to_string(h_tokens[i]) << " " << h_offsets[i];
    if (h_tokens[i] == cudf::io::json::token_t::FieldNameBegin ||
        h_tokens[i] == cudf::io::json::token_t::StringBegin ||
        h_tokens[i] == cudf::io::json::token_t::ValueBegin) {
      str_begin = h_offsets[i];
    }
    if (h_tokens[i] == cudf::io::json::token_t::FieldNameEnd ||
        h_tokens[i] == cudf::io::json::token_t::StringEnd) {
      uint32_t str_end = h_offsets[i];
      // strings are inclusive, but include the quotes
      std::string d(&h_str_data[str_begin + 1], str_end - str_begin - 1);
      ss << " |" << d << "|";
    }
    if (h_tokens[i] == cudf::io::json::token_t::ValueEnd) {
      uint32_t str_end = h_offsets[i];
      // value end is not inclusive
      std::string d(&h_str_data[str_begin], str_end - str_begin);
      ss << " |" << d << "|";
    }

    if (separator.size() > 0 && i + 1 < h_tokens.size()) { ss << separator; }
  }
  std::cerr << ss.str() << std::endl;
}

std::unique_ptr<cudf::column> is_empty_or_null(cudf::column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto byte_count =
    cudf::strings::count_bytes(cudf::strings_column_view{input}, mr);  // stream not exposed yet...
  using IntScalarType = cudf::scalar_type_t<int32_t>;
  auto zero = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::INT32}, stream, mr);
  reinterpret_cast<IntScalarType*>(zero.get())->set_value(0, stream);
  zero->set_valid_async(true, stream);
  auto is_empty             = cudf::binary_operation(*byte_count,
                                         *zero,
                                         cudf::binary_operator::LESS_EQUAL,
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream,
                                         mr);
  auto is_null              = cudf::is_null(input, stream, mr);
  auto mostly_empty_or_null = cudf::binary_operation(*is_empty,
                                                     *is_null,
                                                     cudf::binary_operator::NULL_LOGICAL_OR,
                                                     cudf::data_type{cudf::type_id::BOOL8},
                                                     stream,
                                                     mr);
  is_empty.reset();
  is_null.reset();
  zero.reset();
  auto null_lit    = cudf::make_string_scalar("null", stream, mr);
  auto is_lit_null = cudf::binary_operation(*null_lit,
                                            input,
                                            cudf::binary_operator::EQUAL,
                                            cudf::data_type{cudf::type_id::BOOL8},
                                            stream,
                                            mr);
  return cudf::binary_operation(*is_lit_null,
                                *mostly_empty_or_null,
                                cudf::binary_operator::NULL_LOGICAL_OR,
                                cudf::data_type{cudf::type_id::BOOL8},
                                stream,
                                mr);
}

bool contains_char(cudf::column_view const& input,
                   std::string const& needle,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  cudf::string_scalar s(needle, true, stream, mr);
  auto has_s           = cudf::strings::contains(cudf::strings_column_view(input), s);
  auto any             = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto ret             = cudf::reduce(*has_s,
                          *any,
                          cudf::data_type{cudf::type_id::BOOL8},
                          stream,
                          mr);  // no stream is supported for reduce yet
  using BoolScalarType = cudf::scalar_type_t<bool>;
  return ret->is_valid(stream) && reinterpret_cast<BoolScalarType*>(ret.get())->value(stream);
}

rmm::device_buffer extract_character_buffer(std::unique_ptr<cudf::column> input,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  return std::move(*(input->release().data.release()));
  // Sadly there is no good way around this. We have to make a copy of the data...
  // cudf::strings_column_view scv(input);
  // auto data_length = scv.chars_size(stream);
  // rmm::device_uvector<cudf::io::json::SymbolT> ret(data_length, stream, mr);
  // CUDF_CUDA_TRY(cudaMemcpyAsync(ret.data(),
  //                               scv.chars_begin(stream),
  //                               data_length,
  //                               cudaMemcpyDefault,
  //                               stream.value()));
  // return ret;
}

rmm::device_buffer just_concat(cudf::column_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const input_scv = cudf::strings_column_view{input};

  auto constexpr num_levels = 256;
  rmm::device_uvector<uint32_t> d_histogram(
    num_levels, stream);  // TODO fix it with custom kernel, to avoid overflow to zero. atomicOr?
  thrust::fill(rmm::exec_policy(stream), d_histogram.begin(), d_histogram.end(), 0);
  auto lower_level = std::numeric_limits<char>::min();
  auto upper_level = std::numeric_limits<char>::max();

  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(nullptr,
                                      temp_storage_bytes,
                                      input_scv.chars_begin(stream),
                                      d_histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      input_scv.chars_size(stream),
                                      stream.value());
  rmm::device_buffer d_temp(temp_storage_bytes, stream);
  cub::DeviceHistogram::HistogramEven(d_temp.data(),
                                      temp_storage_bytes,
                                      input_scv.chars_begin(stream),
                                      d_histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      input_scv.chars_size(stream),
                                      stream.value());
  auto zero_level = d_histogram.begin() - lower_level;
  auto first_non_existing_char =
    thrust::find(rmm::exec_policy(stream), zero_level + '\n', d_histogram.end(), 0) - zero_level;
  std::cout << "found first non-existing char:" << first_non_existing_char << std::endl;
  // TODO return this, so that we can configure json reader for delimiter.
  if (first_non_existing_char != '\n') {
    throw std::logic_error(
      "line separator is present in input string, can't process with JSON reader");
  }

  auto first_char = *thrust::device_pointer_cast(input_scv.chars_begin(stream));
  auto all_done   = cudf::strings::join_strings(
    input_scv,
    cudf::string_scalar(std::string(1, first_non_existing_char), true, stream, mr),
    cudf::string_scalar(first_char == '[' ? "[]" : "{}", true, stream, mr),
    stream,
    mr);
  return extract_character_buffer(std::move(all_done), stream, mr);
}

std::pair<rmm::device_buffer, std::unique_ptr<cudf::column>> clean_and_concat(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const input_scv = cudf::strings_column_view{input};
  auto stripped        = cudf::strings::strip(input_scv,
                                       cudf::strings::side_type::BOTH,
                                       cudf::string_scalar("", true, stream, mr),
                                       stream,
                                       mr);
  auto is_n_or_e       = is_empty_or_null(*stripped, stream, mr);
  auto empty_row       = cudf::make_string_scalar("{}", stream, mr);
  auto cleaned         = cudf::copy_if_else(*empty_row, *stripped, *is_n_or_e, stream, mr);
  stripped.reset();
  empty_row.reset();
  if (contains_char(*cleaned, "\n", stream, mr)) {
    throw std::logic_error("line separator is not currently supported in a JSON string");
  }
  if (contains_char(*cleaned, "\r", stream, mr)) {
    throw std::logic_error("carriage return is not currently supported in a JSON string");
  }
  // Eventually we want to use null, but for now...
  auto all_done = cudf::strings::join_strings(
    cudf::strings_column_view(*cleaned),
    cudf::string_scalar("\n", true, stream, mr),
    cudf::string_scalar("{}", true, stream, mr),  // This should be ignored
    stream,
    mr);
  return std::make_pair(extract_character_buffer(std::move(all_done), stream, mr),
                        std::move(is_n_or_e));
}

/// Extracts paths from vector of string columns returned by libcudf json reader
std::vector<std::unique_ptr<cudf::column>> extract_result_columns(
  cudf::io::table_with_metadata&& result,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // vector of columns. inside it could be struct or lists.
  CUDF_FUNC_RANGE();

  auto num_rows  = result.tbl->num_rows();
  auto root      = cudf::make_structs_column(num_rows, (result.tbl)->release(), 0, {});
  auto& metadata = result.metadata.schema_info;
  std::vector<std::unique_ptr<cudf::column>> output;
  output.reserve(json_paths.size());
  for (auto path : json_paths) {
    auto* this_meta = &metadata;
    auto this_col   = root->view();
    std::unique_ptr<cudf::column> holder;
    bool notfound = false;
    bool is_root  = true;
    for (auto [path_type, name, index] : path) {
      // std::cout<<name<<","<<index<<".";
      bool is_list_root = is_root && path_type == path_instruction_type::INDEX;
      if (is_list_root or (path_type == path_instruction_type::NAMED)) {
        auto col_name = is_list_root ? std::to_string(index) : name;
        auto it = std::find_if(this_meta->begin(), this_meta->end(), [col_name](auto col_info) {
          return col_info.name == col_name;
        });
        if (it == this_meta->end()) {
          // NOT FOUND, null column or nullptr?
          // output.push_back(nullptr);
          std::cout << "struct " << col_name << " not found\n";
          std::cout << "struct:\n";
          std::for_each(
            this_meta->begin(), this_meta->end(), [](auto c) { std::cout << c.name << ","; });
          std::cout << "\n";
          notfound = true;
          break;
        } else {
          // next.
          // if (it->children.empty()) {
          //   // output.push_back(std::make_unique<cudf::column>(this_col.child(it -
          //   this_meta->begin()))); break;
          // }
          this_col  = this_col.child(it - this_meta->begin());
          this_meta = &(it->children);
        }
      } else if (path_type == path_instruction_type::WILDCARD) {
        // std::cout<<"list:";
        // std::for_each(this_meta->begin(), this_meta->end(), [](auto c) { std::cout<<c.name<<",";
        // }); std::cout<<"\n";
        if (this_meta->empty()) {
          std::cout << "list * not found\n";
          notfound = true;
          break;
        }
        this_meta = &((*this_meta)[1].children);
        this_col  = this_col.child(1);
      } else if (path_type == path_instruction_type::INDEX) {
        // std::cout<<"list:";
        // std::for_each(this_meta->begin(), this_meta->end(), [](auto c) { std::cout<<c.name<<",";
        // }); std::cout<<"\n";
        if (this_meta->empty()) {
          std::cout << "list " << index << " not found\n";
          notfound = true;
          break;
        }
        this_meta = &((*this_meta)[1].children);
        holder    = cudf::lists::extract_list_element(this_col, index, stream, mr);
        this_col  = holder->view();
      } else {
        CUDF_FAIL("Invalid path instruction type");
      }
      is_root = false;
    }
    // std::cout<<"\n";
    if (notfound)
      output.push_back(make_strings_column(
        num_rows,
        make_column_from_scalar(cudf::numeric_scalar<cudf::size_type>(0), num_rows + 1, stream, mr),
        rmm::device_buffer{},
        num_rows,
        cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr)));
    else
      output.emplace_back(std::make_unique<cudf::column>(this_col, stream, mr));
  }
  return output;
}

// std::map<std::string, cudf::io::schema_element>
auto json_path_to_schema(
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths)
{
  cudf::io::schema_element root;
  for (auto const& instructions : json_paths) {
    cudf::io::schema_element* this_schema = &root;
    bool is_root                          = true;
    for (auto const& [type, name, index] : instructions) {
      if (type == path_instruction_type::INDEX or type == path_instruction_type::WILDCARD) {
        // std::cout << "[";
        this_schema->type = cudf::data_type{cudf::type_id::LIST};
        if (is_root and type == path_instruction_type::INDEX) {
          this_schema =
            &(this_schema
                ->child_types[std::to_string(index)]);  // JSON parser implementation specific
          // std::cout<<index;
        } else {
          this_schema = &(this_schema->child_types["element"]);  // TODO consider using index here
          // std::cout<<"*";
        }
        // std::cout<<"]";
        this_schema->type = cudf::data_type{cudf::type_id::STRING};
      } else if (type == path_instruction_type::NAMED) {
        // std::cout<<"."<<name;
        this_schema->type = cudf::data_type{cudf::type_id::STRUCT};
        this_schema       = &(this_schema->child_types[name]);
        this_schema->type = cudf::data_type{cudf::type_id::STRING};
      } else {
        CUDF_FAIL("Invalid path instruction type");
      }
      is_root = false;
    }
  }
  return root;
  // return std::move(root.child_types);
}

std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths2(
  cudf::column_view const& input,
  std::vector<json_path_t> const& json_paths,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Invalid input format");

  auto const num_outputs = json_paths.size();
  // std::cout<<"num_outputs:"<<num_outputs<<"\n";
  // Input is empty or all nulls - just return all null columns.
  if (input.is_empty() || input.size() == input.null_count()) {
    std::vector<std::unique_ptr<cudf::column>> output;
    for (std::size_t idx = 0; idx < num_outputs; ++idx) {
      output.emplace_back(std::make_unique<cudf::column>(input, stream, mr));
    }
    return output;
  }
  // auto [cleaned, was_empty] = clean_and_concat(input, stream, mr);
  auto cleaned = just_concat(input, stream, mr);
  // print_debug<char, char>(cleaned, "CLEANED INPUT", "", stream);
  cudf::io::datasource::owning_buffer<rmm::device_buffer> buffer{std::move(cleaned)};
  cudf::io::json::detail::normalize_single_quotes(buffer, stream, mr);
  // print_debug<char, char>(buffer, "QUOTE NORMALIZED", "", stream);
  // cleaned = cudf::io::json::detail::normalize_whitespace(std::move(cleaned), stream, mr);
  // print_debug<char, char>(cleaned, "WS NORMALIZED", "", stream);
  //  We will probably do ws normalization as we write out the data. This is true for number
  //  normalization too

  // auto schema = json_path_to_schema(json_paths);
  // print_schema(schema);
  auto json_opts = cudf::io::json_reader_options_builder(
                     cudf::io::source_info(cudf::device_span<std::byte const>{
                       reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}))
                     .lines(true)
                     .mixed_types_as_string(true)
                     .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                     .dtypes(json_paths)
                    //  .dtypes(schema.child_types)
                     .prune_columns(true)
                     .strict_validation(true)
                     .build();

  // auto const [tokens, token_indices] = cudf::io::json::detail::get_token_stream(
  //   cudf::device_span<char const>{reinterpret_cast<char const*>(buffer.data()), buffer.size()},
  //   json_opts,
  //   stream,
  //   mr);
  auto result = cudf::io::read_json(json_opts, stream, mr);
  // return extract_result_columns(std::move(result), json_paths, stream, mr);
  return result.tbl->release();
}

std::vector<json_path_t> pathstrs_to_json_paths(std::vector<std::string> const& paths)
{
  std::vector<json_path_t> json_paths;
  const std::string delims = ".[";
  json_paths.reserve(paths.size());
  for (std::string_view strpath : paths) {
    size_t start = 0;
    json_path_t jpath;
    while (start < strpath.size()) {
      size_t end = strpath.find_first_of(delims, start);
      std::string_view this_path;
      if (end == std::string_view::npos) {
        this_path = strpath.substr(start);
      } else {
        this_path = strpath.substr(start, end - start);
        start     = end + 1;
      }
      if (this_path == "$")
        continue;
      else if (this_path == "*]") {
        jpath.emplace_back(path_instruction_type::WILDCARD, "", -1);
      } else if (this_path.back() == ']') {
        auto index = std::stoi(std::string(this_path.substr(0, this_path.size() - 1)));
        jpath.emplace_back(path_instruction_type::INDEX, "", index);
      } else {
        jpath.emplace_back(path_instruction_type::NAMED, this_path, -1);
      }
      if (end == std::string_view::npos) break;
    }
    json_paths.push_back(jpath);
  }
  return json_paths;
}

}  // namespace io
}  // namespace cudf
