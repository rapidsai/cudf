/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <strings/utilities.cuh>
#include <strings/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Function logic for the replace API.
 *
 * This will perform a replace operation on each string.
 */
struct replace_fn {
  column_device_view const d_strings;
  string_view const d_target;
  string_view const d_repl;
  int32_t const max_repl;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    const char* in_ptr = d_str.data();

    char* out_ptr = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto max_n    = (max_repl < 0) ? d_str.length() : max_repl;
    auto bytes    = d_str.size_bytes();
    auto position = d_str.find(d_target);

    size_type last_pos = 0;
    while ((position >= 0) && (max_n > 0)) {
      if (out_ptr) {
        auto const curr_pos = d_str.byte_offset(position);
        out_ptr = copy_and_increment(out_ptr, in_ptr + last_pos, curr_pos - last_pos);  // copy left
        out_ptr = copy_string(out_ptr, d_repl);                                         // copy repl
        last_pos = curr_pos + d_target.size_bytes();
      } else {
        bytes += d_repl.size_bytes() - d_target.size_bytes();
      }
      position = d_str.find(d_target, position + d_target.size_bytes());
      --max_n;
    }
    if (out_ptr)  // copy whats left (or right depending on your point of view)
      memcpy(out_ptr, in_ptr + last_pos, d_str.size_bytes() - last_pos);
    else
      d_offsets[idx] = bytes;
  }
};

}  // namespace

//
std::unique_ptr<column> replace(strings_column_view const& strings,
                                string_scalar const& target,
                                string_scalar const& repl,
                                int32_t maxrepl,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  CUDF_EXPECTS(repl.is_valid(), "Parameter repl must be valid.");
  CUDF_EXPECTS(target.is_valid(), "Parameter target must be valid.");
  CUDF_EXPECTS(target.size() > 0, "Parameter target must not be empty string.");

  string_view d_target(target.data(), target.size());
  string_view d_repl(repl.data(), repl.size());

  auto d_strings = column_device_view::create(strings.parent(), stream);

  // this utility calls the given functor to build the offsets and chars columns
  auto children =
    cudf::strings::detail::make_strings_children(replace_fn{*d_strings, d_target, d_repl, maxrepl},
                                                 strings.size(),
                                                 strings.null_count(),
                                                 stream,
                                                 mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

namespace {
/**
 * @brief Function logic for the replace_slice API.
 *
 * This will perform a replace_slice operation on each string.
 */
struct replace_slice_fn {
  column_device_view const d_strings;
  string_view const d_repl;
  size_type const start;
  size_type const stop;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    auto const length  = d_str.length();
    char const* in_ptr = d_str.data();
    auto const begin   = d_str.byte_offset(((start < 0) || (start > length) ? length : start));
    auto const end     = d_str.byte_offset(((stop < 0) || (stop > length) ? length : stop));

    if (d_chars) {
      char* out_ptr = d_chars + d_offsets[idx];

      out_ptr = copy_and_increment(out_ptr, in_ptr, begin);  // copy beginning
      out_ptr = copy_string(out_ptr, d_repl);                // insert replacement
      out_ptr = copy_and_increment(out_ptr,                  // copy end
                                   in_ptr + end,
                                   d_str.size_bytes() - end);
    } else {
      d_offsets[idx] = d_str.size_bytes() + d_repl.size_bytes() - (end - begin);
    }
  }
};

}  // namespace

std::unique_ptr<column> replace_slice(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      size_type start,
                                      size_type stop,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  CUDF_EXPECTS(repl.is_valid(), "Parameter repl must be valid.");
  if (stop > 0) CUDF_EXPECTS(start <= stop, "Parameter start must be less than or equal to stop.");

  string_view d_repl(repl.data(), repl.size());

  auto d_strings = column_device_view::create(strings.parent(), stream);

  // this utility calls the given functor to build the offsets and chars columns
  auto children =
    cudf::strings::detail::make_strings_children(replace_slice_fn{*d_strings, d_repl, start, stop},
                                                 strings.size(),
                                                 strings.null_count(),
                                                 stream,
                                                 mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

namespace {
/**
 * @brief Function logic for the replace_multi API.
 *
 * This will perform the multi-replace operation on each string.
 */
struct replace_multi_fn {
  column_device_view const d_strings;
  column_device_view const d_targets;
  column_device_view const d_repls;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    char const* in_ptr = d_str.data();

    size_type bytes = d_str.size_bytes();
    size_type spos  = 0;
    size_type lpos  = 0;
    char* out_ptr   = d_chars ? d_chars + d_offsets[idx] : nullptr;

    // check each character against each target
    while (spos < d_str.size_bytes()) {
      for (int tgt_idx = 0; tgt_idx < d_targets.size(); ++tgt_idx) {
        auto const d_tgt = d_targets.element<string_view>(tgt_idx);
        if ((d_tgt.size_bytes() <= (d_str.size_bytes() - spos)) &&    // check fit
            (d_tgt.compare(in_ptr + spos, d_tgt.size_bytes()) == 0))  // and match
        {
          auto const d_repl = (d_repls.size() == 1) ? d_repls.element<string_view>(0)
                                                    : d_repls.element<string_view>(tgt_idx);
          bytes += d_repl.size_bytes() - d_tgt.size_bytes();
          if (out_ptr) {
            out_ptr = copy_and_increment(out_ptr, in_ptr + lpos, spos - lpos);
            out_ptr = copy_string(out_ptr, d_repl);
            lpos    = spos + d_tgt.size_bytes();
          }
          spos += d_tgt.size_bytes() - 1;
          break;
        }
      }
      ++spos;
    }
    if (out_ptr)  // copy remainder
      memcpy(out_ptr, in_ptr + lpos, d_str.size_bytes() - lpos);
    else
      d_offsets[idx] = bytes;
  }
};

}  // namespace

std::unique_ptr<column> replace(strings_column_view const& strings,
                                strings_column_view const& targets,
                                strings_column_view const& repls,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  CUDF_EXPECTS(((targets.size() > 0) && (targets.null_count() == 0)),
               "Parameters targets must not be empty and must not have nulls");
  CUDF_EXPECTS(((repls.size() > 0) && (repls.null_count() == 0)),
               "Parameters repls must not be empty and must not have nulls");
  if (repls.size() > 1)
    CUDF_EXPECTS(repls.size() == targets.size(), "Sizes for targets and repls must match");

  auto d_strings = column_device_view::create(strings.parent(), stream);
  auto d_targets = column_device_view::create(targets.parent(), stream);
  auto d_repls   = column_device_view::create(repls.parent(), stream);

  // this utility calls the given functor to build the offsets and chars columns
  auto children =
    cudf::strings::detail::make_strings_children(replace_multi_fn{*d_strings, *d_targets, *d_repls},
                                                 strings.size(),
                                                 strings.null_count(),
                                                 stream,
                                                 mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

std::unique_ptr<column> replace_nulls(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_strings_column(stream, mr);
  CUDF_EXPECTS(repl.is_valid(), "Parameter repl must be valid.");

  string_view d_repl(repl.data(), repl.size());

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // build offsets column
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int32_t>(0), [d_strings, d_repl] __device__(size_type idx) {
      return d_strings.is_null(idx) ? d_repl.size_bytes()
                                    : d_strings.element<string_view>(idx).size_bytes();
    });
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // build chars column
  size_type bytes   = thrust::device_pointer_cast(d_offsets)[strings_count];
  auto chars_column = strings::detail::create_chars_child_column(
    strings_count, strings.null_count(), bytes, stream, mr);
  auto d_chars = chars_column->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     [d_strings, d_repl, d_offsets, d_chars] __device__(size_type idx) {
                       string_view d_str = d_repl;
                       if (!d_strings.is_null(idx)) d_str = d_strings.element<string_view>(idx);
                       memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
                     });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             0,
                             rmm::device_buffer{0, stream, mr},
                             stream,
                             mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> replace(strings_column_view const& strings,
                                string_scalar const& target,
                                string_scalar const& repl,
                                int32_t maxrepl,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace(strings, target, repl, maxrepl, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> replace_slice(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      size_type start,
                                      size_type stop,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_slice(strings, repl, start, stop, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> replace(strings_column_view const& strings,
                                strings_column_view const& targets,
                                strings_column_view const& repls,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace(strings, targets, repls, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> replace_nulls(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nulls(strings, repl, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
