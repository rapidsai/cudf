/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/cache.hpp"
#include "regex_ir.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/experimental/strings/regex.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/utility>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudf::experimental {
namespace {

using pair_t = cuda::std::pair<char const*, size_type>;

static_assert(sizeof(pair_t) == 16);
static_assert(alignof(pair_t) == alignof(char const*));

// A single large block limits resident warps enough to retain each thread's sequential input
// window in L1. Smaller blocks over-occupy these memory-bound kernels and thrash that window.
constexpr std::uint32_t block_size        = 1024;
constexpr std::size_t required_stack_size = 64U * 1024U;

struct input_data {
  char const* chars;
  void const* offsets;
  bitmask_type const* validity;
  size_type row_offset;
  size_type rows;
  bool offset64;
};

input_data get_input_data(strings_column_view const& input, rmm::cuda_stream_view stream)
{
  auto const offsets = input.offsets();
  CUDF_EXPECTS(offsets.type().id() == type_id::INT32 || offsets.type().id() == type_id::INT64,
               "Unsupported strings offset type");
  return input_data{input.chars_begin(stream),
                    offsets.type().id() == type_id::INT64
                      ? static_cast<void const*>(offsets.data<std::int64_t>())
                      : static_cast<void const*>(offsets.data<std::int32_t>()),
                    input.parent().null_mask(),
                    input.offset(),
                    input.size(),
                    offsets.type().id() == type_id::INT64};
}

std::string normalize_pattern(std::string_view pattern)
{
  std::string result;
  result.reserve(pattern.size());
  for (std::size_t position = 0; position < pattern.size();) {
    if (pattern[position] != '\\' || position + 1 == pattern.size()) {
      result.push_back(pattern[position++]);
      continue;
    }

    auto const escaped = pattern[position + 1];
    if (escaped < '0' || escaped > '7') {
      result.append(pattern.substr(position, 2));
      position += 2;
      continue;
    }

    position += 1;
    std::uint32_t value = 0;
    std::size_t digits  = 0;
    while (position < pattern.size() && digits < 3 && pattern[position] >= '0' &&
           pattern[position] <= '7') {
      value = (value << 3U) | static_cast<std::uint32_t>(pattern[position] - '0');
      ++position;
      ++digits;
    }
    result += value <= 0xff ? std::format("\\x{:02X}", value) : std::format("\\u{:04X}", value);
  }
  return result;
}

regex_ir::compile_options make_compile_options(strings::regex_flags flags)
{
  regex_ir::compile_options options;
  options.case_insensitive = strings::is_ignorecase(flags);
  options.multiline        = strings::is_multiline(flags);
  options.dot_all          = strings::is_dotall(flags);
  options.ascii_classes    = strings::is_ascii(flags);
  options.extended_newline = strings::is_ext_newline(flags);
  return options;
}

void ensure_stack_size()
{
  std::size_t stack_size = 0;
  CUDF_CUDA_TRY(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
  if (stack_size < required_stack_size) {
    CUDF_CUDA_TRY(cudaDeviceSetLimit(cudaLimitStackSize, required_stack_size));
  }
}

kernel prepare_kernel(std::string_view pattern,
                      strings::regex_flags flags,
                      regex_ir::operation_kind operation,
                      std::optional<std::string> replacement,
                      std::string wrapper,
                      std::string_view name)
{
  ensure_stack_size();
  auto matcher = regex_ir::compile(
    normalize_pattern(pattern), operation, std::move(replacement), make_compile_options(flags));
  auto module   = regex_ir::nvvm::assemble(std::move(matcher), std::move(wrapper));
  auto fragment = get_nvvm_fragment(std::string{name}, module);
  rtcx::memory_fragment fragments[] = {
    {.data = fragment->view(), .type = rtcx::binary_type::LTO_IR, .name = nullptr}};
  return get_lto_linked_kernel(std::string{name}, {}, fragments);
}

void launch(kernel& prepared,
            input_data const& input,
            rmm::cuda_stream_view stream,
            void* output,
            void const* extra = nullptr)
{
  if (input.rows == 0) { return; }
  auto chars      = const_cast<char*>(input.chars);
  auto offsets    = const_cast<void*>(input.offsets);
  auto validity   = const_cast<bitmask_type*>(input.validity);
  auto row_offset = input.row_offset;
  auto rows       = input.rows;
  auto grid =
    static_cast<std::uint32_t>((static_cast<std::uint32_t>(rows) + block_size - 1) / block_size);
  if (extra == nullptr) {
    prepared.launch_with({grid, 1, 1},
                         {block_size, 1, 1},
                         0,
                         stream,
                         chars,
                         offsets,
                         validity,
                         row_offset,
                         rows,
                         output);
  } else {
    prepared.launch_with({grid, 1, 1},
                         {block_size, 1, 1},
                         0,
                         stream,
                         chars,
                         offsets,
                         validity,
                         row_offset,
                         rows,
                         output,
                         extra);
  }
}

std::unique_ptr<column> fixed_result(strings_column_view const& input,
                                     std::string_view pattern,
                                     strings::regex_flags flags,
                                     regex_ir::operation_kind operation,
                                     data_type output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto result = make_numeric_column(output_type,
                                    input.size(),
                                    cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                    input.null_count(),
                                    stream,
                                    mr);
  if (input.is_empty()) { return result; }
  auto data     = get_input_data(input, stream);
  auto prepared = prepare_kernel(pattern,
                                 flags,
                                 operation,
                                 std::nullopt,
                                 regex_ir::nvvm::make_fixed_kernel(data.offset64, operation),
                                 "cudf.experimental.regex.fixed");
  launch(prepared, data, stream, result->mutable_view().head<void>());
  return result;
}

std::unique_ptr<strings::regex_program> make_capture_program(std::string_view pattern,
                                                             strings::regex_flags flags)
{
  return strings::regex_program::create(pattern, flags, strings::capture_groups::EXTRACT);
}

std::unique_ptr<column> make_strings(rmm::device_uvector<pair_t> const& pairs,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  return cudf::make_strings_column(
    device_span<pair_t const>{pairs.data(), pairs.size()}, stream, mr);
}

std::unique_ptr<table> extract_impl(strings_column_view const& input,
                                    std::string_view pattern,
                                    strings::regex_flags flags,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  auto program = make_capture_program(pattern, flags);
  auto groups  = program->groups_count();
  CUDF_EXPECTS(groups > 0, "Group indicators not found in regex pattern");

  if (input.is_empty()) {
    std::vector<std::unique_ptr<column>> columns;
    columns.reserve(groups);
    for (size_type group = 0; group < groups; ++group) {
      columns.emplace_back(make_empty_column(type_id::STRING));
    }
    return std::make_unique<table>(std::move(columns));
  }

  auto pair_count = static_cast<std::size_t>(input.size()) * static_cast<std::size_t>(groups);
  rmm::device_uvector<pair_t> pairs(pair_count, stream, mr);
  auto data     = get_input_data(input, stream);
  auto prepared = prepare_kernel(
    pattern,
    flags,
    regex_ir::operation_kind::EXTRACT,
    std::nullopt,
    regex_ir::nvvm::make_capture_kernel(data.offset64, (groups + 1) * 2, 0, groups, true),
    "cudf.experimental.regex.extract");
  launch(prepared, data, stream, pairs.data());

  std::vector<device_span<pair_t const>> spans;
  spans.reserve(groups);
  for (size_type group = 0; group < groups; ++group) {
    spans.emplace_back(pairs.data() + static_cast<std::size_t>(group) * input.size(), input.size());
  }
  auto columns = cudf::make_strings_column_batch(spans, stream, mr);
  return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<column> extract_single_impl(strings_column_view const& input,
                                            std::string_view pattern,
                                            size_type group,
                                            strings::regex_flags flags,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  auto program = make_capture_program(pattern, flags);
  auto groups  = program->groups_count();
  CUDF_EXPECTS(groups > 0, "capture groups not found in regex pattern", std::invalid_argument);
  CUDF_EXPECTS(group >= 0 && group < groups,
               "group parameter outside the range of capture groups found in the regex pattern",
               std::invalid_argument);

  rmm::device_uvector<pair_t> pairs(input.size(), stream, mr);
  auto data     = get_input_data(input, stream);
  auto prepared = prepare_kernel(
    pattern,
    flags,
    regex_ir::operation_kind::EXTRACT,
    std::nullopt,
    regex_ir::nvvm::make_capture_kernel(data.offset64, (groups + 1) * 2, group, 1, false),
    "cudf.experimental.regex.extract_single");
  launch(prepared, data, stream, pairs.data());
  return make_strings(pairs, stream, mr);
}

struct offsets_result {
  std::unique_ptr<column> offsets;
  size_type total;
};

offsets_result sizes_to_offsets(column_view const& sizes,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto zero =
    make_numeric_column(data_type{type_id::INT32}, 1, mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(zero->mutable_view().data<size_type>(), 0, sizeof(size_type), stream.value()));
  if (sizes.is_empty()) { return {std::move(zero), 0}; }

  auto sum    = make_sum_aggregation<scan_aggregation>();
  auto prefix = cudf::scan(sizes, *sum, scan_type::INCLUSIVE, null_policy::EXCLUDE, stream, mr);
  std::array<column_view, 2> pieces{zero->view(), prefix->view()};
  auto offsets      = cudf::concatenate(pieces, stream, mr);
  auto total_scalar = cudf::get_element(prefix->view(), prefix->size() - 1, stream, mr);
  auto total        = static_cast<numeric_scalar<size_type> const&>(*total_scalar).value(stream);
  return {std::move(offsets), total};
}

std::unique_ptr<column> enumerate_impl(strings_column_view const& input,
                                       std::string_view pattern,
                                       strings::regex_flags flags,
                                       strings::capture_groups captures,
                                       bool findall,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto program       = make_capture_program(pattern, flags);
  auto capture_count = program->groups_count();
  auto groups        = captures == strings::capture_groups::EXTRACT ? capture_count : 0;
  if (findall) {
    CUDF_EXPECTS(groups <= 1, "findall does not support more than 1 capture group");
  } else {
    CUDF_EXPECTS(groups > 0, "extract_all requires group indicators in the regex pattern.");
  }
  if (input.is_empty()) { return make_empty_lists_column(data_type{type_id::STRING}); }

  auto counts = make_numeric_column(
    data_type{type_id::INT32}, input.size(), mask_state::UNALLOCATED, stream, mr);
  auto validity = make_numeric_column(
    data_type{type_id::BOOL8}, input.size(), mask_state::UNALLOCATED, stream, mr);
  auto data        = get_input_data(input, stream);
  auto slots       = (capture_count + 1) * 2;
  auto size_kernel = prepare_kernel(pattern,
                                    flags,
                                    regex_ir::operation_kind::EXTRACT,
                                    std::nullopt,
                                    regex_ir::nvvm::make_enumeration_size_kernel(
                                      data.offset64, slots, findall ? 1 : groups, !findall),
                                    "cudf.experimental.regex.enumerate_size");
  launch(size_kernel,
         data,
         stream,
         counts->mutable_view().head<void>(),
         validity->mutable_view().head<void>());

  auto offsets = sizes_to_offsets(counts->view(), stream, mr);
  rmm::device_uvector<pair_t> pairs(offsets.total, stream, mr);
  if (offsets.total > 0) {
    auto emit_kernel = prepare_kernel(
      pattern,
      flags,
      regex_ir::operation_kind::EXTRACT,
      std::nullopt,
      regex_ir::nvvm::make_enumeration_emit_kernel(data.offset64, slots, groups, findall),
      "cudf.experimental.regex.enumerate_emit");
    launch(emit_kernel, data, stream, pairs.data(), offsets.offsets->view().data<size_type>());
  }
  auto strings_output = make_strings(pairs, stream, mr);

  rmm::device_buffer null_mask;
  size_type null_count;
  if (findall) {
    null_mask  = cudf::detail::copy_bitmask(input.parent(), stream, mr);
    null_count = input.null_count();
  } else {
    auto converted = cudf::bools_to_mask(validity->view(), stream, mr);
    null_mask      = std::move(*converted.first);
    null_count     = converted.second;
  }
  return make_lists_column(input.size(),
                           std::move(offsets.offsets),
                           std::move(strings_output),
                           null_count,
                           std::move(null_mask));
}

using replacement_piece = regex_ir::nvvm::replacement_piece;

std::vector<replacement_piece> literal_replacement(std::string_view replacement)
{
  return {{std::string{replacement}, std::nullopt}};
}

std::vector<replacement_piece> parse_backref_replacement(std::string_view replacement,
                                                         size_type group_count)
{
  CUDF_EXPECTS(!replacement.empty(), "Parameter replacement must not be empty");
  auto uses_backslash = false;
  for (std::size_t index = 0; index + 1 < replacement.size(); ++index) {
    if (replacement[index] == '\\' &&
        std::isdigit(static_cast<unsigned char>(replacement[index + 1])) != 0) {
      uses_backslash = true;
      break;
    }
  }

  std::vector<replacement_piece> result;
  std::string literal;
  auto flush_literal = [&] {
    if (!literal.empty()) {
      result.push_back({std::move(literal), std::nullopt});
      literal.clear();
    }
  };
  for (std::size_t position = 0; position < replacement.size();) {
    auto capture_begin = std::string_view::npos;
    auto capture_end   = std::string_view::npos;
    if (uses_backslash && replacement[position] == '\\' && position + 1 < replacement.size() &&
        std::isdigit(static_cast<unsigned char>(replacement[position + 1])) != 0) {
      capture_begin = position + 1;
      capture_end   = capture_begin;
      while (capture_end < replacement.size() &&
             std::isdigit(static_cast<unsigned char>(replacement[capture_end])) != 0) {
        ++capture_end;
      }
    } else if (!uses_backslash && replacement[position] == '$' &&
               position + 3 < replacement.size() && replacement[position + 1] == '{' &&
               std::isdigit(static_cast<unsigned char>(replacement[position + 2])) != 0) {
      capture_begin = position + 2;
      capture_end   = capture_begin;
      while (capture_end < replacement.size() &&
             std::isdigit(static_cast<unsigned char>(replacement[capture_end])) != 0) {
        ++capture_end;
      }
      if (capture_end == replacement.size() || replacement[capture_end] != '}') {
        capture_begin = std::string_view::npos;
      }
    }

    if (capture_begin == std::string_view::npos) {
      literal.push_back(replacement[position++]);
      continue;
    }

    std::uint64_t capture = 0;
    for (auto index = capture_begin; index < capture_end; ++index) {
      capture = capture * 10U + static_cast<std::uint64_t>(replacement[index] - '0');
    }
    CUDF_EXPECTS(capture <= static_cast<std::uint64_t>(std::min(group_count, size_type{99})),
                 "Group index numbers must be in the range 0 to group count");
    flush_literal();
    result.push_back({{}, static_cast<size_type>(capture)});
    position = uses_backslash ? capture_end : capture_end + 1;
  }
  flush_literal();
  return result;
}

std::unique_ptr<column> replace_impl(strings_column_view const& input,
                                     std::string_view pattern,
                                     std::vector<replacement_piece> replacement,
                                     std::optional<size_type> max_replace_count,
                                     strings::regex_flags flags,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  if (pattern.empty()) { return std::make_unique<column>(input.parent(), stream, mr); }
  if (max_replace_count.has_value() && *max_replace_count == 0) {
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  auto limit =
    max_replace_count.has_value() && *max_replace_count > 0 ? max_replace_count : std::nullopt;
  auto const native_replacement = !limit.has_value();
  auto const replacement_template =
    native_replacement ? std::optional<std::string>{regex_ir::nvvm::encode_replacement(replacement)}
                       : std::nullopt;
  size_type capture_slots = 0;
  if (!native_replacement) {
    auto program  = make_capture_program(pattern, flags);
    capture_slots = (program->groups_count() + 1) * 2;
  }

  auto sizes = make_numeric_column(
    data_type{type_id::INT32}, input.size(), mask_state::UNALLOCATED, stream, mr);
  auto data        = get_input_data(input, stream);
  auto size_kernel = prepare_kernel(
    pattern,
    flags,
    native_replacement ? regex_ir::operation_kind::REPLACE : regex_ir::operation_kind::EXTRACT,
    replacement_template,
    native_replacement ? regex_ir::nvvm::make_replace_kernel(data.offset64, false)
                       : regex_ir::nvvm::make_limited_replace_kernel(
                           data.offset64, false, replacement, capture_slots, *limit),
    "cudf.experimental.regex.replace_size");
  launch(size_kernel, data, stream, sizes->mutable_view().head<void>());
  auto offsets = sizes_to_offsets(sizes->view(), stream, mr);
  rmm::device_buffer chars(offsets.total, stream, mr);
  if (offsets.total > 0) {
    auto emit_kernel = prepare_kernel(
      pattern,
      flags,
      native_replacement ? regex_ir::operation_kind::REPLACE : regex_ir::operation_kind::EXTRACT,
      replacement_template,
      native_replacement ? regex_ir::nvvm::make_replace_kernel(data.offset64, true)
                         : regex_ir::nvvm::make_limited_replace_kernel(
                             data.offset64, true, replacement, capture_slots, *limit),
      "cudf.experimental.regex.replace_emit");
    launch(emit_kernel, data, stream, chars.data(), offsets.offsets->view().data<size_type>());
  }
  return make_strings_column(input.size(),
                             std::move(offsets.offsets),
                             std::move(chars),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

struct split_result {
  std::unique_ptr<column> lists;
  size_type columns;
};

split_result split_record_impl(strings_column_view const& input,
                               std::string_view pattern,
                               size_type maxsplit,
                               bool reverse,
                               strings::regex_flags flags,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");
  if (input.is_empty()) { return {make_empty_lists_column(data_type{type_id::STRING}), 1}; }

  auto effective_counts = make_numeric_column(
    data_type{type_id::INT32}, input.size(), mask_state::UNALLOCATED, stream, mr);
  auto full_counts = make_numeric_column(
    data_type{type_id::INT32}, input.size(), mask_state::UNALLOCATED, stream, mr);
  auto data        = get_input_data(input, stream);
  auto size_kernel = prepare_kernel(pattern,
                                    flags,
                                    regex_ir::operation_kind::SPLIT,
                                    std::nullopt,
                                    regex_ir::nvvm::make_split_size_kernel(data.offset64, maxsplit),
                                    "cudf.experimental.regex.split_size");
  launch(size_kernel,
         data,
         stream,
         effective_counts->mutable_view().head<void>(),
         full_counts->mutable_view().head<void>());

  auto maximum = cudf::reduce(effective_counts->view(),
                              *make_max_aggregation<reduce_aggregation>(),
                              data_type{type_id::INT32},
                              stream,
                              mr);
  auto columns = static_cast<numeric_scalar<size_type> const&>(*maximum).value(stream);
  columns      = std::max(columns, size_type{1});

  auto effective_offsets = sizes_to_offsets(effective_counts->view(), stream, mr);
  auto full_offsets      = sizes_to_offsets(full_counts->view(), stream, mr);
  rmm::device_uvector<pair_t> pairs(effective_offsets.total, stream, mr);
  rmm::device_uvector<std::int64_t> spans(
    static_cast<std::size_t>(full_offsets.total) * 2, stream, mr);
  if (effective_offsets.total > 0) {
    auto emit_kernel =
      prepare_kernel(pattern,
                     flags,
                     regex_ir::operation_kind::SPLIT,
                     std::nullopt,
                     regex_ir::nvvm::make_split_emit_kernel(data.offset64, reverse),
                     "cudf.experimental.regex.split_emit");
    auto chars      = const_cast<char*>(data.chars);
    auto offsets    = const_cast<void*>(data.offsets);
    auto validity   = const_cast<bitmask_type*>(data.validity);
    auto row_offset = data.row_offset;
    auto rows       = data.rows;
    auto grid =
      static_cast<std::uint32_t>((static_cast<std::uint32_t>(rows) + block_size - 1) / block_size);
    auto effective_data = effective_offsets.offsets->view().data<size_type>();
    auto full_data      = full_offsets.offsets->view().data<size_type>();
    emit_kernel.launch_with({grid, 1, 1},
                            {block_size, 1, 1},
                            0,
                            stream,
                            chars,
                            offsets,
                            validity,
                            row_offset,
                            rows,
                            pairs.data(),
                            effective_data,
                            full_data,
                            spans.data());
  }
  auto strings_output = make_strings(pairs, stream, mr);
  auto lists          = make_lists_column(input.size(),
                                 std::move(effective_offsets.offsets),
                                 std::move(strings_output),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr));
  return {std::move(lists), columns};
}

std::unique_ptr<table> split_table_impl(strings_column_view const& input,
                                        std::string_view pattern,
                                        size_type maxsplit,
                                        bool reverse,
                                        strings::regex_flags flags,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    std::vector<std::unique_ptr<column>> columns;
    columns.emplace_back(make_empty_column(type_id::STRING));
    return std::make_unique<table>(std::move(columns));
  }
  auto result = split_record_impl(input, pattern, maxsplit, reverse, flags, stream, mr);
  lists_column_view lists{result.lists->view()};
  std::vector<std::unique_ptr<column>> columns;
  columns.reserve(result.columns);
  for (size_type index = 0; index < result.columns; ++index) {
    auto column = cudf::lists::extract_list_element(lists, index, stream, mr);
    if (column->null_count() == 0) { column->set_null_mask(rmm::device_buffer{}, 0); }
    columns.emplace_back(std::move(column));
  }
  return std::make_unique<table>(std::move(columns));
}

}  // namespace

std::unique_ptr<column> contains_re_jit(strings_column_view const& input,
                                        std::string_view pattern,
                                        strings::regex_flags flags,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  return fixed_result(input,
                      pattern,
                      flags,
                      regex_ir::operation_kind::CONTAINS,
                      data_type{type_id::BOOL8},
                      stream,
                      mr);
}

std::unique_ptr<column> matches_re_jit(strings_column_view const& input,
                                       std::string_view pattern,
                                       strings::regex_flags flags,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto anchored = std::format(R"(\A(?:{}))", pattern);
  return fixed_result(input,
                      anchored,
                      flags,
                      regex_ir::operation_kind::CONTAINS,
                      data_type{type_id::BOOL8},
                      stream,
                      mr);
}

std::unique_ptr<column> count_re_jit(strings_column_view const& input,
                                     std::string_view pattern,
                                     strings::regex_flags flags,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  return fixed_result(
    input, pattern, flags, regex_ir::operation_kind::COUNT, data_type{type_id::INT32}, stream, mr);
}

std::unique_ptr<column> find_re_jit(strings_column_view const& input,
                                    std::string_view pattern,
                                    strings::regex_flags flags,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  return fixed_result(
    input, pattern, flags, regex_ir::operation_kind::FIND, data_type{type_id::INT32}, stream, mr);
}

std::unique_ptr<table> extract_jit(strings_column_view const& input,
                                   std::string_view pattern,
                                   strings::regex_flags flags,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  return extract_impl(input, pattern, flags, stream, mr);
}

std::unique_ptr<column> extract_single_jit(strings_column_view const& input,
                                           std::string_view pattern,
                                           size_type group,
                                           strings::regex_flags flags,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  return extract_single_impl(input, pattern, group, flags, stream, mr);
}

std::unique_ptr<column> extract_all_record_jit(strings_column_view const& input,
                                               std::string_view pattern,
                                               strings::regex_flags flags,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return enumerate_impl(input, pattern, flags, strings::capture_groups::EXTRACT, false, stream, mr);
}

std::unique_ptr<column> findall_jit(strings_column_view const& input,
                                    std::string_view pattern,
                                    strings::regex_flags flags,
                                    strings::capture_groups captures,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  return enumerate_impl(input, pattern, flags, captures, true, stream, mr);
}

std::unique_ptr<column> replace_re_jit(strings_column_view const& input,
                                       std::string_view pattern,
                                       string_scalar const& replacement,
                                       std::optional<size_type> max_replace_count,
                                       strings::regex_flags flags,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  return replace_impl(input,
                      pattern,
                      literal_replacement(replacement.to_string(stream)),
                      max_replace_count,
                      flags,
                      stream,
                      mr);
}

std::unique_ptr<column> replace_with_backrefs_jit(strings_column_view const& input,
                                                  std::string_view pattern,
                                                  std::string_view replacement,
                                                  strings::regex_flags flags,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto program = make_capture_program(pattern, flags);
  CUDF_EXPECTS(!pattern.empty(), "Parameter pattern must not be empty");
  return replace_impl(input,
                      pattern,
                      parse_backref_replacement(replacement, program->groups_count()),
                      std::nullopt,
                      flags,
                      stream,
                      mr);
}

std::unique_ptr<table> split_re_jit(strings_column_view const& input,
                                    std::string_view pattern,
                                    size_type maxsplit,
                                    strings::regex_flags flags,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  return split_table_impl(input, pattern, maxsplit, false, flags, stream, mr);
}

std::unique_ptr<table> rsplit_re_jit(strings_column_view const& input,
                                     std::string_view pattern,
                                     size_type maxsplit,
                                     strings::regex_flags flags,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  return split_table_impl(input, pattern, maxsplit, true, flags, stream, mr);
}

std::unique_ptr<column> split_record_re_jit(strings_column_view const& input,
                                            std::string_view pattern,
                                            size_type maxsplit,
                                            strings::regex_flags flags,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  return split_record_impl(input, pattern, maxsplit, false, flags, stream, mr).lists;
}

std::unique_ptr<column> rsplit_record_re_jit(strings_column_view const& input,
                                             std::string_view pattern,
                                             size_type maxsplit,
                                             strings::regex_flags flags,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  return split_record_impl(input, pattern, maxsplit, true, flags, stream, mr).lists;
}

}  // namespace cudf::experimental
