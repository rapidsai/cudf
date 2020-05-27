/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Utility to return integer column indicating the postion of
 * target string within each string in a strings column.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam FindFunction Returns integer character position value given a string and target.
 *
 * @param strings Strings column to search for target.
 * @param target String to search for in each string in the strings column.
 * @param start First character position to start the search.
 * @param stop Last character position (exclusive) to end the search.
 * @param pfn Functor used for locating `target` in each string.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for kernel calls.
 * @return New integer column with character position values.
 */
template <typename FindFunction>
std::unique_ptr<column> find_fn(strings_column_view const& strings,
                                string_scalar const& target,
                                size_type start,
                                size_type stop,
                                FindFunction& pfn,
                                rmm::mr::device_memory_resource* mr,
                                cudaStream_t stream)
{
  CUDF_EXPECTS(target.is_valid(), "Parameter target must be valid.");
  CUDF_EXPECTS(start >= 0, "Parameter start must be positive integer or zero.");
  if ((stop > 0) && (start > stop)) CUDF_FAIL("Parameter start must be less than stop.");
  //
  auto d_target       = string_view(target.data(), target.size());
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  auto strings_count  = strings.size();
  // create output column
  auto results      = make_numeric_column(data_type{INT32},
                                     strings_count,
                                     copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<int32_t>();
  // set the position values by evaluating the passed function
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_results,
                    [d_strings, pfn, d_target, start, stop] __device__(size_type idx) {
                      int32_t position = -1;
                      if (!d_strings.is_null(idx))
                        position = static_cast<int32_t>(
                          pfn(d_strings.element<string_view>(idx), d_target, start, stop));
                      return position;
                    });
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace

std::unique_ptr<column> find(strings_column_view const& strings,
                             string_scalar const& target,
                             size_type start                     = 0,
                             size_type stop                      = -1,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                             cudaStream_t stream                 = 0)
{
  auto pfn = [] __device__(
               string_view d_string, string_view d_target, size_type start, size_type stop) {
    size_type length = d_string.length();
    if (d_target.empty()) return start > length ? -1 : start;
    size_type begin = (start > length) ? length : start;
    size_type end   = (stop < 0) || (stop > length) ? length : stop;
    return d_string.find(d_target, begin, end - begin);
  };

  return find_fn(strings, target, start, stop, pfn, mr, stream);
}

std::unique_ptr<column> rfind(strings_column_view const& strings,
                              string_scalar const& target,
                              size_type start                     = 0,
                              size_type stop                      = -1,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                              cudaStream_t stream                 = 0)
{
  auto pfn = [] __device__(
               string_view d_string, string_view d_target, size_type start, size_type stop) {
    size_type length = d_string.length();
    size_type begin  = (start > length) ? length : start;
    size_type end    = (stop < 0) || (stop > length) ? length : stop;
    if (d_target.empty()) return start > length ? -1 : end;
    return d_string.rfind(d_target, begin, end - begin);
  };

  return find_fn(strings, target, start, stop, pfn, mr, stream);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> find(strings_column_view const& strings,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::find(strings, target, start, stop, mr);
}

std::unique_ptr<column> rfind(strings_column_view const& strings,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::rfind(strings, target, start, stop, mr);
}

namespace detail {
namespace {
/**
 * @brief Utility to return a bool column indicating the presence of
 * a given target string in a strings column.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for target.
 * @param target UTF-8 encoded string to check in strings column.
 * @param pfn Returns bool value if target is found in the given string.
 * @param mr Resource for allocating device memory.
 * @param stream Stream to use for kernel calls.
 * @return New BOOL column.
 */
template <typename BoolFunction>
std::unique_ptr<column> contains_fn(strings_column_view const& strings,
                                    string_scalar const& target,
                                    BoolFunction pfn,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_numeric_column(data_type{BOOL8}, 0);

  CUDF_EXPECTS(target.is_valid(), "Parameter target must be valid.");
  if (target.size() == 0)  // empty target string returns true
  {
    auto const true_scalar = make_fixed_width_scalar<bool>(true, stream);
    auto results           = make_column_from_scalar(*true_scalar, strings.size(), mr, stream);
    results->set_null_mask(copy_bitmask(strings.parent(), stream, mr), strings.null_count());
    return results;
  }

  auto d_target       = string_view(target.data(), target.size());
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{BOOL8},
                                     strings_count,
                                     copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    d_results,
    [d_strings, pfn, d_target] __device__(size_type idx) {
      if (!d_strings.is_null(idx))
        return static_cast<bool>(pfn(d_strings.element<string_view>(idx), d_target));
      return false;
    });
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace

std::unique_ptr<column> contains(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return d_string.find(d_target) >= 0;
  };
  return contains_fn(strings, target, pfn, mr, stream);
}

std::unique_ptr<column> starts_with(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return d_string.find(d_target) == 0;
  };
  return contains_fn(strings, target, pfn, mr, stream);
}

std::unique_ptr<column> ends_with(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    auto str_length = d_string.length();
    auto tgt_length = d_target.length();
    if (str_length < tgt_length) return false;
    return d_string.find(d_target, str_length - tgt_length) >= 0;
  };

  return contains_fn(strings, target, pfn, mr, stream);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 string_scalar const& target,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, target, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, target, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, target, mr);
}

// For substring_index APIs
namespace detail {
// Internal helper class
namespace {

struct substring_index_functor {
  template <typename ColItrT, typename DelimiterItrT>
  std::unique_ptr<column> operator()(ColItrT const col_itr,
                                     DelimiterItrT const delim_itr,
                                     size_type delimiter_count,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream,
                                     size_type strings_count) const
  {
    // Shallow copy of the resultant strings
    rmm::device_vector<string_view> out_col_strings(strings_count);

    // Invalid output column strings - null rows
    string_view const invalid_str{nullptr, 0};

    thrust::transform(
      rmm::exec_policy(stream)->on(stream),
      col_itr,
      col_itr + strings_count,
      delim_itr,
      out_col_strings.data().get(),
      [delimiter_count, invalid_str] __device__(auto col_val_pair, auto delim_val_pair) {
        // If the column value for this row or the delimiter is null or if the delimiter count is 0,
        // result is null
        if (!col_val_pair.second || !delim_val_pair.second || delimiter_count == 0)
          return invalid_str;
        auto col_val = col_val_pair.first;

        // If the global delimiter or the row specific delimiter or if the column value for the row
        // is empty, value is empty.
        if (delim_val_pair.first.empty() || col_val.empty()) return string_view{};

        auto delim_val = delim_val_pair.first;

        auto const col_val_len   = col_val.length();
        auto const delimiter_len = delim_val.length();

        auto nsearches      = (delimiter_count < 0) ? -delimiter_count : delimiter_count;
        size_type start_pos = 0;
        size_type end_pos   = col_val_len;
        string_view out_str{};

        for (auto i = 0; i < nsearches; ++i) {
          if (delimiter_count < 0) {
            end_pos = col_val.rfind(delim_val, 0, end_pos);
            if (end_pos == -1) {
              out_str = col_val;
              break;
            }
            if (i + 1 == nsearches)
              out_str =
                col_val.substr(end_pos + delimiter_len, col_val_len - end_pos - delimiter_len);
          } else {
            auto char_pos = col_val.find(delim_val, start_pos);
            if (char_pos == -1) {
              out_str = col_val;
              break;
            }
            if (i + 1 == nsearches)
              out_str = col_val.substr(0, char_pos);
            else
              start_pos = char_pos + delimiter_len;
          }
        }

        return out_str.empty() ? string_view{} : out_str;
      });

    // Create an output column with the resultant strings
    return make_strings_column(out_col_strings, invalid_str, stream, mr);
  }
};

}  // namespace

template <typename DelimiterItrT>
std::unique_ptr<column> substring_index(strings_column_view const& strings,
                                        DelimiterItrT const delimiter_itr,
                                        size_type count,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream = 0)
{
  auto strings_count = strings.size();
  // If there aren't any rows, return an empty strings column
  if (strings_count == 0) return strings::detail::make_empty_strings_column(mr, stream);

  // Create device view of the column
  auto colview_ptr = column_device_view::create(strings.parent(), stream);
  auto colview     = *colview_ptr;
  if (colview.nullable()) {
    return substring_index_functor{}(
      cudf::detail::make_pair_iterator<string_view, true>(colview),
      delimiter_itr,
      count,
      mr,
      stream,
      strings_count);
  } else {
    return substring_index_functor{}(
      cudf::detail::make_pair_iterator<string_view, false>(colview),
      delimiter_itr,
      count,
      mr,
      stream,
      strings_count);
  }
}

}  // namespace detail

// external APIs

std::unique_ptr<column> substring_index(strings_column_view const& strings,
                                        string_scalar const& delimiter,
                                        size_type count,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::substring_index(
    strings, cudf::detail::make_pair_iterator<string_view>(delimiter), count, mr);
}

std::unique_ptr<column> substring_index(strings_column_view const& strings,
                                        strings_column_view const& delimiters,
                                        size_type count,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(strings.size() == delimiters.size(),
               "Strings and delimiters column sizes do not match");

  CUDF_FUNC_RANGE();
  auto delimiters_dev_view_ptr = cudf::column_device_view::create(delimiters.parent(), 0);
  auto delimiters_dev_view     = *delimiters_dev_view_ptr;
  return (delimiters_dev_view.nullable())
           ? detail::substring_index(
               strings,
               cudf::detail::make_pair_iterator<string_view, true>(delimiters_dev_view),
               count,
               mr)
           : detail::substring_index(
               strings,
               cudf::detail::make_pair_iterator<string_view, false>(delimiters_dev_view),
               count,
               mr);
}

}  // namespace strings
}  // namespace cudf
