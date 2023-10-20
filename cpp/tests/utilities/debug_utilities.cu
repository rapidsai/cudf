/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/detail/get_value.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <iomanip>
#include <sstream>

namespace cudf::test {

// Forward declaration.
namespace detail {

/**
 * @brief Formats a column view as a string
 *
 * @param col The column view
 * @param delimiter The delimiter to put between strings
 * @param indent Indentation for all output
 */
std::string to_string(cudf::column_view const& col,
                      std::string const& delimiter,
                      std::string const& indent = "");

/**
 * @brief Formats a null mask as a string
 *
 * @param null_mask The null mask buffer
 * @param null_mask_size Size of the null mask (in rows)
 * @param indent Indentation for all output
 */
std::string to_string(std::vector<bitmask_type> const& null_mask,
                      size_type null_mask_size,
                      std::string const& indent = "");

/**
 * @brief Convert column values to a host vector of strings
 *
 * Supports indentation of all output.  For example, if the displayed output of your column
 * would be
 *
 * @code{.pseudo}
 * "1,2,3,4,5"
 * @endcode
 * and the `indent` parameter was "   ", that indentation would be prepended to
 * result in the output
 * @code{.pseudo}
 * "   1,2,3,4,5"
 * @endcode
 *
 * The can be useful for displaying complex types. An example use case would be for
 * displaying the nesting of a LIST type column (via recursion).
 *
 *  List<List<int>>:
 *  Length : 3
 *  Offsets : 0, 2, 5, 6
 *  Children :
 *     List<int>:
 *     Length : 6
 *     Offsets : 0, 2, 4, 7, 8, 9, 11
 *     Children :
 *        1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10
 *
 * @param col The column view
 * @param indent Indentation for all output
 */
std::vector<std::string> to_strings(cudf::column_view const& col, std::string const& indent = "");

}  // namespace detail

namespace {

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
static auto numeric_to_string_precise(T value)
{
  return std::to_string(value);
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
static auto numeric_to_string_precise(T value)
{
  std::ostringstream o;
  o << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
  return o.str();
}

static auto duration_suffix(cudf::duration_D) { return " days"; }

static auto duration_suffix(cudf::duration_s) { return " seconds"; }

static auto duration_suffix(cudf::duration_ms) { return " milliseconds"; }

static auto duration_suffix(cudf::duration_us) { return " microseconds"; }

static auto duration_suffix(cudf::duration_ns) { return " nanoseconds"; }

std::string get_nested_type_str(cudf::column_view const& view)
{
  if (view.type().id() == cudf::type_id::LIST) {
    lists_column_view lcv(view);
    return cudf::type_to_name(view.type()) + "<" + (get_nested_type_str(lcv.child())) + ">";
  }

  if (view.type().id() == cudf::type_id::STRUCT) {
    std::ostringstream out;

    out << cudf::type_to_name(view.type()) + "<";
    std::transform(view.child_begin(),
                   view.child_end(),
                   std::ostream_iterator<std::string>(out, ","),
                   [&out](auto const col) { return get_nested_type_str(col); });
    out << ">";
    return out.str();
  }

  return cudf::type_to_name(view.type());
}

template <typename NestedColumnView>
std::string nested_offsets_to_string(NestedColumnView const& c, std::string const& delimiter = ", ")
{
  column_view offsets = (c.parent()).child(NestedColumnView::offsets_column_index);
  CUDF_EXPECTS(offsets.type().id() == type_id::INT32,
               "Column does not appear to be an offsets column");
  CUDF_EXPECTS(offsets.offset() == 0, "Offsets column has an internal offset!");
  size_type output_size = c.size() + 1;

  // the first offset value to normalize everything against
  size_type first =
    cudf::detail::get_value<size_type>(offsets, c.offset(), cudf::get_default_stream());
  rmm::device_uvector<size_type> shifted_offsets(output_size, cudf::get_default_stream());

  // normalize the offset values for the column offset
  size_type const* d_offsets = offsets.head<size_type>() + c.offset();
  thrust::transform(
    rmm::exec_policy(cudf::get_default_stream()),
    d_offsets,
    d_offsets + output_size,
    shifted_offsets.begin(),
    [first] __device__(int32_t offset) { return static_cast<size_type>(offset - first); });

  auto const h_shifted_offsets =
    cudf::detail::make_host_vector_sync(shifted_offsets, cudf::get_default_stream());
  std::ostringstream buffer;
  for (size_t idx = 0; idx < h_shifted_offsets.size(); idx++) {
    buffer << h_shifted_offsets[idx];
    if (idx < h_shifted_offsets.size() - 1) { buffer << delimiter; }
  }
  return buffer.str();
}

struct column_view_printer {
  template <typename Element, std::enable_if_t<is_numeric<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
  {
    auto h_data = cudf::test::to_host<Element>(col);

    out.resize(col.size());

    if (col.nullable()) {
      std::transform(thrust::make_counting_iterator(size_type{0}),
                     thrust::make_counting_iterator(col.size()),
                     out.begin(),
                     [&h_data](auto idx) {
                       return bit_is_set(h_data.second.data(), idx)
                                ? numeric_to_string_precise(h_data.first[idx])
                                : std::string("NULL");
                     });

    } else {
      std::transform(h_data.first.begin(), h_data.first.end(), out.begin(), [](Element el) {
        return numeric_to_string_precise(el);
      });
    }
  }

  template <typename Element, std::enable_if_t<is_timestamp<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    //  For timestamps, convert timestamp column to column of strings, then
    //  call string version
    std::string format = [&]() {
      if constexpr (std::is_same_v<cudf::timestamp_s, Element>) {
        return std::string{"%Y-%m-%dT%H:%M:%SZ"};
      } else if constexpr (std::is_same_v<cudf::timestamp_ms, Element>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%3fZ"};
      } else if constexpr (std::is_same_v<cudf::timestamp_us, Element>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%6fZ"};
      } else if constexpr (std::is_same_v<cudf::timestamp_ns, Element>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%9fZ"};
      }
      return std::string{"%Y-%m-%d"};
    }();

    auto col_as_strings = cudf::strings::from_timestamps(col, format);
    if (col_as_strings->size() == 0) { return; }

    this->template operator()<cudf::string_view>(*col_as_strings, out, indent);
  }

  template <typename Element, std::enable_if_t<cudf::is_fixed_point<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
  {
    auto const h_data = cudf::test::to_host<Element>(col);
    if (col.nullable()) {
      std::transform(thrust::make_counting_iterator(size_type{0}),
                     thrust::make_counting_iterator(col.size()),
                     std::back_inserter(out),
                     [&h_data](auto idx) {
                       return h_data.second.empty() || bit_is_set(h_data.second.data(), idx)
                                ? static_cast<std::string>(h_data.first[idx])
                                : std::string("NULL");
                     });
    } else {
      std::transform(std::cbegin(h_data.first),
                     std::cend(h_data.first),
                     std::back_inserter(out),
                     [col](auto const& fp) { return static_cast<std::string>(fp); });
    }
  }

  template <typename Element,
            std::enable_if_t<std::is_same_v<Element, cudf::string_view>>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
  {
    //
    //  Implementation for strings, call special to_host variant
    //
    if (col.is_empty()) return;
    auto h_data = cudf::test::to_host<std::string>(col);

    // explicitly replace some special whitespace characters with their literal equivalents
    auto cleaned = [](std::string_view in) {
      std::string out(in);
      auto replace_char = [](std::string& out, char c, std::string_view repl) {
        for (std::string::size_type pos{}; out.npos != (pos = out.find(c, pos)); pos++) {
          out.replace(pos, 1, repl);
        }
      };
      replace_char(out, '\a', "\\a");
      replace_char(out, '\b', "\\b");
      replace_char(out, '\f', "\\f");
      replace_char(out, '\r', "\\r");
      replace_char(out, '\t', "\\t");
      replace_char(out, '\n', "\\n");
      replace_char(out, '\v', "\\v");
      return out;
    };

    out.resize(col.size());
    std::transform(thrust::make_counting_iterator(size_type{0}),
                   thrust::make_counting_iterator(col.size()),
                   out.begin(),
                   [&](auto idx) {
                     return h_data.second.empty() || bit_is_set(h_data.second.data(), idx)
                              ? cleaned(h_data.first[idx])
                              : std::string("NULL");
                   });
  }

  template <typename Element,
            std::enable_if_t<std::is_same_v<Element, cudf::dictionary32>>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
  {
    cudf::dictionary_column_view dictionary(col);
    if (col.is_empty()) return;
    std::vector<std::string> keys    = to_strings(dictionary.keys());
    std::vector<std::string> indices = to_strings({dictionary.indices().type(),
                                                   dictionary.size(),
                                                   dictionary.indices().head(),
                                                   dictionary.null_mask(),
                                                   dictionary.null_count(),
                                                   dictionary.offset()});
    out.insert(out.end(), keys.begin(), keys.end());
    if (!indices.empty()) {
      std::string first = "\x08 : " + indices.front();  // use : as delimiter
      out.push_back(first);                             // between keys and indices
      out.insert(out.end(), indices.begin() + 1, indices.end());
    }
  }

  // Print the tick counts with the units
  template <typename Element, std::enable_if_t<is_duration<Element>()>* = nullptr>
  void operator()(cudf::column_view const& col, std::vector<std::string>& out, std::string const&)
  {
    auto h_data = cudf::test::to_host<Element>(col);

    out.resize(col.size());

    if (col.nullable()) {
      std::transform(thrust::make_counting_iterator(size_type{0}),
                     thrust::make_counting_iterator(col.size()),
                     out.begin(),
                     [&h_data](auto idx) {
                       return bit_is_set(h_data.second.data(), idx)
                                ? numeric_to_string_precise(h_data.first[idx].count()) +
                                    duration_suffix(h_data.first[idx])
                                : std::string("NULL");
                     });

    } else {
      std::transform(h_data.first.begin(), h_data.first.end(), out.begin(), [](Element el) {
        return numeric_to_string_precise(el.count()) + duration_suffix(el);
      });
    }
  }

  template <typename Element, std::enable_if_t<std::is_same_v<Element, cudf::list_view>>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    lists_column_view lcv(col);

    // propagate slicing to the child if necessary
    column_view child    = lcv.get_sliced_child(cudf::get_default_stream());
    bool const is_sliced = lcv.offset() > 0 || child.offset() > 0;

    std::string tmp =
      get_nested_type_str(col) + (is_sliced ? "(sliced)" : "") + ":\n" + indent +
      "Length : " + std::to_string(lcv.size()) + "\n" + indent +
      "Offsets : " + (lcv.size() > 0 ? nested_offsets_to_string(lcv) : "") + "\n" +
      (lcv.parent().nullable()
         ? indent + "Null count: " + std::to_string(lcv.null_count()) + "\n" +
             detail::to_string(cudf::test::bitmask_to_host(col), col.size(), indent) + "\n"
         : "") +
      // non-nested types don't typically display their null masks, so do it here for convenience.
      (!is_nested(child.type()) && child.nullable()
         ? "   " + detail::to_string(cudf::test::bitmask_to_host(child), child.size(), indent) +
             "\n"
         : "") +
      (detail::to_string(child, ", ", indent + "   ")) + "\n";

    out.push_back(tmp);
  }

  template <typename Element,
            std::enable_if_t<std::is_same_v<Element, cudf::struct_view>>* = nullptr>
  void operator()(cudf::column_view const& col,
                  std::vector<std::string>& out,
                  std::string const& indent)
  {
    structs_column_view view{col};

    std::ostringstream out_stream;

    out_stream << get_nested_type_str(col) << ":\n"
               << indent << "Length : " << view.size() << ":\n";
    if (view.nullable()) {
      out_stream << indent << "Null count: " << view.null_count() << "\n"
                 << detail::to_string(cudf::test::bitmask_to_host(col), col.size(), indent) << "\n";
    }

    auto iter = thrust::make_counting_iterator(0);
    std::transform(
      iter,
      iter + view.num_children(),
      std::ostream_iterator<std::string>(out_stream, "\n"),
      [&](size_type index) {
        auto child = view.get_sliced_child(index, cudf::get_default_stream());

        // non-nested types don't typically display their null masks, so do it here for convenience.
        return (!is_nested(child.type()) && child.nullable()
                  ? "   " +
                      detail::to_string(cudf::test::bitmask_to_host(child), child.size(), indent) +
                      "\n"
                  : "") +
               detail::to_string(child, ", ", indent + "   ");
      });

    out.push_back(out_stream.str());
  }
};

}  // namespace

namespace detail {

/**
 * @copydoc cudf::test::detail::to_strings
 */
std::vector<std::string> to_strings(cudf::column_view const& col, std::string const& indent)
{
  std::vector<std::string> reply;
  cudf::type_dispatcher(col.type(), column_view_printer{}, col, reply, indent);
  return reply;
}

/**
 * @copydoc cudf::test::detail::to_string(cudf::column_view, std::string, std::string)
 *
 * @param indent Indentation for all output
 */
std::string to_string(cudf::column_view const& col,
                      std::string const& delimiter,
                      std::string const& indent)
{
  std::ostringstream buffer;
  std::vector<std::string> h_data = to_strings(col, indent);

  buffer << indent;
  std::copy(h_data.begin(),
            h_data.end() - (!h_data.empty()),
            std::ostream_iterator<std::string>(buffer, delimiter.c_str()));
  if (!h_data.empty()) buffer << h_data.back();

  return buffer.str();
}

/**
 * @copydoc cudf::test::detail::to_string(std::vector<bitmask_type>, size_type, std::string)
 *
 * @param indent Indentation for all output.  See comment in `to_strings` for
 * a detailed description.
 */
std::string to_string(std::vector<bitmask_type> const& null_mask,
                      size_type null_mask_size,
                      std::string const& indent)
{
  std::ostringstream buffer;
  buffer << indent;
  for (int idx = null_mask_size - 1; idx >= 0; idx--) {
    buffer << (cudf::bit_is_set(null_mask.data(), idx) ? "1" : "0");
  }
  return buffer.str();
}

}  // namespace detail

std::vector<std::string> to_strings(cudf::column_view const& col)
{
  return detail::to_strings(col);
}

std::string to_string(cudf::column_view const& col, std::string const& delimiter)
{
  return detail::to_string(col, delimiter);
}

std::string to_string(std::vector<bitmask_type> const& null_mask, size_type null_mask_size)
{
  return detail::to_string(null_mask, null_mask_size);
}

void print(cudf::column_view const& col, std::ostream& os)
{
  os << to_string(col, ",") << std::endl;
}

}  // namespace cudf::test
