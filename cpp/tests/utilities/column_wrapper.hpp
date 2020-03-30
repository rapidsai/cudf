/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_buffer.hpp>
#include <tests/utilities/cudf_gtest.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <iterator>
#include <memory>

namespace cudf {
namespace test {
/**---------------------------------------------------------------------------*
 * @brief Convenience wrapper for creating a `thrust::transform_iterator` over a
 * `thrust::counting_iterator`.
 *
 * Example:
 * ```
 * // Returns square of the value of the counting iterator
 * auto iter = make_counting_transform_iterator(0, [](auto i){ return (i * i);}); 
 * iter[0] == 0 
 * iter[1] == 1 
 * iter[2] == 4
 * ...
 * iter[n] == n * n
 * ```
 *
 * @param start The starting value of the counting iterator
 * @param f The unary function to apply to the counting iterator.
 * This should be a host function and not a device function.
 * @return auto A transform iterator that applies `f` to a counting iterator
 *---------------------------------------------------------------------------**/
template <typename UnaryFunction>
auto make_counting_transform_iterator(cudf::size_type start, UnaryFunction f) {
  return thrust::make_transform_iterator(thrust::make_counting_iterator(start),
                                         f);
}

namespace detail {
/**---------------------------------------------------------------------------*
 * @brief Base class for a wrapper around a `cudf::column`.
 *
 * Classes that derive from `column_wrapper` may be passed directly into any
 * API expecting a `column_view` or `mutable_column_view`.
 *
 * `column_wrapper` should not be instantiated directly.
 *---------------------------------------------------------------------------**/
class column_wrapper {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Implicit conversion operator to `column_view`.
   *
   * Allows passing in a `column_wrapper` (or any class deriving from
   * `column_wrapper`) to be passed into any API expecting a `column_view`
   * parameter.
   *---------------------------------------------------------------------------**/
  operator column_view() const { return wrapped->view(); }

  /**---------------------------------------------------------------------------*
   * @brief Implicit conversion operator to `mutable_column_view`.
   *
   * Allows passing in a `column_wrapper` (or any class deriving from
   * `column_wrapper`) to be passed into any API expecting a
   * `mutable_column_view` parameter.
   *---------------------------------------------------------------------------**/
  operator mutable_column_view() { return wrapped->mutable_view(); }

  /**---------------------------------------------------------------------------*
   * @brief Releases internal unique_ptr to wrapped column
   *---------------------------------------------------------------------------**/
  std::unique_ptr<cudf::column> release() { return std::move(wrapped); }

 protected:
  std::unique_ptr<cudf::column> wrapped{};  ///< The wrapped column
};

/**---------------------------------------------------------------------------*
 * @brief Creates a `device_buffer` containing the elements in the range
 * `[begin,end)`.
 * 
 *
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @param begin Begining of the sequence of elements
 * @param end End of the sequence of elements
 * @return rmm::device_buffer Buffer containing all elements in the range
 *`[begin,end)`
 *---------------------------------------------------------------------------**/
template <typename Element, typename InputIterator>
rmm::device_buffer make_elements(InputIterator begin, InputIterator end) {
  static_assert(cudf::is_fixed_width<Element>(),
                "Unexpected non-fixed width type.");
  std::vector<Element> elements(begin, end);
  return rmm::device_buffer{elements.data(), elements.size() * sizeof(Element)};
}

/**---------------------------------------------------------------------------*
 * @brief Create a `std::vector` containing a validity indicator bitmask using
 * the range `[begin,end)` interpreted as booleans to indicate the state of
 * each bit.
 *
 * If `*(begin + i) == true`, then bit `i` is set to 1, else it is zero.
 *
 * @tparam ValidityIterator
 * @param begin The beginning of the validity indicator sequence
 * @param end The end of the validity indicator sequence
 * @return std::vector Contains a bitmask where bits are set for every
 * element in `[begin,end)` that evaluated to `true`.
 *---------------------------------------------------------------------------**/
template <typename ValidityIterator>
std::vector<bitmask_type> make_null_mask_vector(ValidityIterator begin,
                                                ValidityIterator end) {
  cudf::size_type size = std::distance(begin, end);
  auto num_words =
      cudf::bitmask_allocation_size_bytes(size) / sizeof(bitmask_type);
  std::vector<bitmask_type> null_mask(num_words, 0);
  for (auto i = 0; i < size; ++i) {
    if (begin[i] == true) {
      set_bit_unsafe(null_mask.data(), i);
    }
  }
  return null_mask;
}

/**---------------------------------------------------------------------------*
 * @brief Create a `device_buffer` containing a validity indicator bitmask using
 * the range `[begin,end)` interpreted as booleans to indicate the state of
 *each bit.
 *
 * If `*(begin + i) == true`, then bit `i` is set to 1, else it is zero.
 *
 * @tparam ValidityIterator
 * @param begin The beginning of the validity indicator sequence
 * @param end The end of the validity indicator sequence
 * @return rmm::device_buffer Contains a bitmask where bits are set for every
 * element in `[begin,end)` that evaluated to `true`.
 *---------------------------------------------------------------------------**/
template <typename ValidityIterator>
rmm::device_buffer make_null_mask(ValidityIterator begin,
                                  ValidityIterator end) {
  auto null_mask = make_null_mask_vector(begin, end);
  return rmm::device_buffer{
      null_mask.data(), null_mask.size() * sizeof(decltype(null_mask.front()))};
}

/**---------------------------------------------------------------------------*
 * @brief Given the range `[begin,end)`, converts each value to a string and
 * then creates a packed vector of characters for each string and a vector of
 * offsets indicating the starting position of each string.
 *
 * @tparam StringsIterator A `std::string` must be constructible from
 * dereferencing a `StringsIterator`.
 * @tparam ValidityIterator Dereferencing a ValidityIterator must be
 * convertible to `bool`
 * @param begin The beginning of the sequence of values to convert to strings
 * @param end The end of the sequence of values to convert to strings
 * @param v The beginning of the validity indicator sequence
 * @return std::pair containing the vector of chars and offsets
 *---------------------------------------------------------------------------**/
template <typename StringsIterator, typename ValidityIterator>
auto make_chars_and_offsets(StringsIterator begin, StringsIterator end,
                            ValidityIterator v) {
  std::vector<char> chars{};
  std::vector<int32_t> offsets(1, 0);
  for( auto str = begin; str < end; ++str) {
    std::string tmp = (*v++) ? std::string(*str) : std::string{};
    chars.insert(chars.end(), std::cbegin(tmp), std::cend(tmp));
    offsets.push_back(offsets.back() + tmp.length());
  }
  return std::make_pair(std::move(chars), std::move(offsets));
};
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief `column_wrapper` derived class for wrapping columns of fixed-width
 * elements.
 *
 * @tparam Element The fixed-width element type
 *---------------------------------------------------------------------------**/
template <typename Element>
class fixed_width_column_wrapper : public detail::column_wrapper {
  static_assert(cudf::is_fixed_width<Element>(),
                "Unexpected non-fixed width type.");

 public:
  /**---------------------------------------------------------------------------*
   * @brief Default constructor initializes an empty column with proper dtype
   *---------------------------------------------------------------------------**/
  fixed_width_column_wrapper() : column_wrapper{} {
    std::vector<Element> empty;
    wrapped.reset(new cudf::column{
        cudf::data_type{cudf::experimental::type_to_id<Element>()},
        0,
        detail::make_elements<Element>(empty.begin(), empty.end())});
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a non-nullable column of the fixed-width elements in the
   * range `[begin,end)`.
   *
   * Example:
   * ```c++
   * // Creates a non-nullable column of INT32 elements with 5 elements: {0, 2, 4, 6, 8}
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i*2;});
   * fixed_width_column_wrapper<int32_t> w(elements, elements + 5);
   * ```
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   *---------------------------------------------------------------------------**/
  template <typename InputIterator>
  fixed_width_column_wrapper(InputIterator begin, InputIterator end)
      : column_wrapper{} {
    cudf::size_type size = std::distance(begin, end);
    wrapped.reset(new cudf::column{
        cudf::data_type{cudf::experimental::type_to_id<Element>()}, size,
        detail::make_elements<Element>(begin, end)});
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a nullable column of the fixed-width elements in the range
   * `[begin,end)` using the range `[v, v + distance(begin,end))` interpreted
   * as booleans to indicate the validity of each element.
   *
   * If `v[i] == true`, element `i` is valid, else it is null.
   * 
   * Example:
   * ```c++
   * // Creates a nullable column of INT32 elements with 5 elements: {null, 1, null, 3, null}
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i;});
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;})
   * fixed_width_column_wrapper<int32_t> w(elements, elements + 5, validity);
   * ```
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param v The beginning of the sequence of validity indicators
   *---------------------------------------------------------------------------**/
  template <typename InputIterator, typename ValidityIterator>
  fixed_width_column_wrapper(InputIterator begin, InputIterator end,
                             ValidityIterator v)
      : column_wrapper{} {
    cudf::size_type size = std::distance(begin, end);

    wrapped.reset(new cudf::column{
        cudf::data_type{cudf::experimental::type_to_id<Element>()}, size,
        detail::make_elements<Element>(begin, end),
        detail::make_null_mask(v, v + size), cudf::UNKNOWN_NULL_COUNT});
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a non-nullable column of fixed-width elements from an
   * initializer list.
   *
   * Example:
   * ```c++
   * // Creates a non-nullable INT32 column with 4 elements: {1, 2, 3, 4}
   * fixed_width_column_wrapper<int32_t> w{{1, 2, 3, 4}};
   * ```
   *
   * @param element_list The list of elements
   *---------------------------------------------------------------------------**/
  fixed_width_column_wrapper(std::initializer_list<Element> elements)
      : fixed_width_column_wrapper(std::cbegin(elements), std::cend(elements)) {
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a nullable column from a list of fixed-width elements
   * using another list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of
   * booleans where `true` indicates the element is valid, and `false` indicates
   * the element is null.
   *
   * Example:
   * ```c++
   * // Creates a nullable INT32 column with 4 elements: {1, NULL, 3, NULL}
   * fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, {1, 0, 1, 0}};
   * ```
   *
   * @param elements The list of elements
   * @param validity The list of validity indicator booleans
   *---------------------------------------------------------------------------**/
  fixed_width_column_wrapper(std::initializer_list<Element> elements,
                             std::initializer_list<bool> validity)
      : fixed_width_column_wrapper(std::cbegin(elements), std::cend(elements),
                                   std::cbegin(validity)) {}

  /**---------------------------------------------------------------------------*
   * @brief Construct a nullable column from a list of fixed-width elements and
   * the the range `[v, v + element_list.size())` interpreted as booleans to
   * indicate the validity of each element.
   * 
   * Example:
   * ```c++
   * // Creates a nullable INT32 column with 4 elements: {NULL, 1, NULL, 3}
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;})
   * fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, validity}
   * ```
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   * convertible to `bool`
   * @param element_list The list of elements
   * @param v The beginning of the sequence of validity indicators
   *---------------------------------------------------------------------------**/
  template <typename ValidityIterator>
  fixed_width_column_wrapper(std::initializer_list<Element> element_list,
                             ValidityIterator v)
      : fixed_width_column_wrapper(std::cbegin(element_list),
                                   std::cend(element_list), v) {}
};

/**---------------------------------------------------------------------------*
 * @brief `column_wrapper` derived class for wrapping columns of strings.
 *---------------------------------------------------------------------------**/
class strings_column_wrapper : public detail::column_wrapper {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Construct a non-nullable column of strings from the range
   * `[begin,end)`.
   *
   * Values in the sequence `[begin,end)` will each be converted to
   *`std::string` and a column will be created containing all of the strings.
   *
   * Example:
   * ```c++
   * // Creates a non-nullable STRING column with 7 string elements: 
   * // {"", "this", "is", "a", "column", "of", "strings"}
   * std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
   * strings_column_wrapper s(strings.begin(), strings.end());
   * ```
   *
   * @tparam StringsIterator A `std::string` must be constructible from
   * dereferencing a `StringsIterator`.
   * @param begin The beginning of the sequence
   * @param end The end of the sequence
   *---------------------------------------------------------------------------**/
  template <typename StringsIterator>
  strings_column_wrapper(StringsIterator begin, StringsIterator end)
      : column_wrapper{} {
    std::vector<char> chars;
    std::vector<int32_t> offsets;
    auto all_valid =
        make_counting_transform_iterator(0, [](auto i) { return true; });
    std::tie(chars, offsets) =
        detail::make_chars_and_offsets(begin, end, all_valid);
    wrapped = cudf::make_strings_column(chars, offsets);
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a nullable column of strings from the range
   * `[begin,end)` using the range `[v, v + distance(begin,end))` interpreted
   * as booleans to indicate the validity of each string.
   *
   * Values in the sequence `[begin,end)` will each be converted to
   *`std::string` and a column will be created containing all of the strings.
   *
   * If `v[i] == true`, string `i` is valid, else it is null. If a string
   * `*(begin+i)` is null, it's value is ignored and treated as an empty string.
   * 
   * Example:
   * ```c++
   * // Creates a nullable STRING column with 7 string elements: 
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * strings_column_wrapper s(strings.begin(), strings.end(), validity);
   * ```
   *
   * @tparam StringsIterator A `std::string` must be constructible from
   * dereferencing a `StringsIterator`.
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   * convertible to `bool`
   * @param begin The beginning of the sequence
   * @param end The end of the sequence
   * @param v The beginning of the sequence of validity indicators
   *---------------------------------------------------------------------------**/
  template <typename StringsIterator, typename ValidityIterator>
  strings_column_wrapper(StringsIterator begin, StringsIterator end,
                         ValidityIterator v)
      : column_wrapper{} {
    size_type num_strings = std::distance(begin, end);
    std::vector<char> chars;
    std::vector<int32_t> offsets;
    std::tie(chars, offsets) = detail::make_chars_and_offsets(begin, end, v);
    wrapped = cudf::make_strings_column(
        chars, offsets, detail::make_null_mask_vector(v, v + num_strings));
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a non-nullable column of strings from a list of strings.
   * 
   * Example:
   * ```c++
   * // Creates a non-nullable STRING column with 7 string elements: 
   * // {"", "this", "is", "a", "column", "of", "strings"}
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"});
   * ```
   *
   * @param strings The list of strings
   *---------------------------------------------------------------------------**/
  strings_column_wrapper(std::initializer_list<std::string> strings)
      : strings_column_wrapper(std::cbegin(strings), std::cend(strings)) {}

  /**---------------------------------------------------------------------------*
   * @brief Construct a nullable column of strings from a list of strings and
   * the range `[v, v + strings.size())` interpreted as booleans to indicate the
   * validity of each string.
   * 
   * Example:
   * ```c++
   * // Creates a nullable STRING column with 7 string elements: 
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"}, validity);
   * ```
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   * convertible to `bool`
   * @param strings The list of strings
   * @param v The beginning of the sequence of validity indicators
   *---------------------------------------------------------------------------**/
  template <typename ValidityIterator>
  strings_column_wrapper(std::initializer_list<std::string> strings,
                         ValidityIterator v)
      : strings_column_wrapper(std::cbegin(strings), std::cend(strings), v) {}

  /**---------------------------------------------------------------------------*
   * @brief Construct a nullable column of strings from a list of strings and
   * a list of booleans to indicate the validity of each string.
   * 
   * Example:
   * ```c++
   * // Creates a nullable STRING column with 7 string elements: 
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"}, 
   *                          {0,1,0,1,0,1,0});
   * ```
   *
   * @param strings The list of strings
   * @param validity The list of validity indicator booleans
   *---------------------------------------------------------------------------**/
  strings_column_wrapper(std::initializer_list<std::string> strings,
                         std::initializer_list<bool> validity)
      : strings_column_wrapper(std::cbegin(strings), std::cend(strings),
                               std::cbegin(validity)) {}
};

}  // namespace test
}  // namespace cudf
