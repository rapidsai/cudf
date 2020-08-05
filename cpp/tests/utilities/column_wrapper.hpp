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

#include <cudf/concatenate.hpp>
#include <tests/utilities/column_utilities.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include "cudf/fixed_point/fixed_point.hpp"

namespace cudf {
namespace test {
/**
 * @brief Convenience wrapper for creating a `thrust::transform_iterator` over a
 * `thrust::counting_iterator`.
 *
 * Example:
 * @code{.cpp}
 * // Returns square of the value of the counting iterator
 * auto iter = make_counting_transform_iterator(0, [](auto i){ return (i * i);});
 * iter[0] == 0
 * iter[1] == 1
 * iter[2] == 4
 * ...
 * iter[n] == n * n
 * @endcode
 *
 * @param start The starting value of the counting iterator
 * @param f The unary function to apply to the counting iterator.
 * This should be a host function and not a device function.
 * @return auto A transform iterator that applies `f` to a counting iterator
 **/
template <typename UnaryFunction>
auto make_counting_transform_iterator(cudf::size_type start, UnaryFunction f)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator(start), f);
}

namespace detail {
/**
 * @brief Base class for a wrapper around a `cudf::column`.
 *
 * Classes that derive from `column_wrapper` may be passed directly into any
 * API expecting a `column_view` or `mutable_column_view`.
 *
 * `column_wrapper` should not be instantiated directly.
 **/
class column_wrapper {
 public:
  /**
   * @brief Implicit conversion operator to `column_view`.
   *
   * Allows passing in a `column_wrapper` (or any class deriving from
   * `column_wrapper`) to be passed into any API expecting a `column_view`
   * parameter.
   **/
  operator column_view() const { return wrapped->view(); }

  /**
   * @brief Implicit conversion operator to `mutable_column_view`.
   *
   * Allows passing in a `column_wrapper` (or any class deriving from
   * `column_wrapper`) to be passed into any API expecting a
   * `mutable_column_view` parameter.
   **/
  operator mutable_column_view() { return wrapped->mutable_view(); }

  /**
   * @brief Releases internal unique_ptr to wrapped column
   **/
  std::unique_ptr<cudf::column> release() { return std::move(wrapped); }

 protected:
  std::unique_ptr<cudf::column> wrapped{};  ///< The wrapped column
};

/**
 * @brief Convert between source and target types when they differ and where possible.
 **/
template <typename From, typename To>
struct fixed_width_type_converter {
  // Are the types same - simply copy elements from [begin, end) to out
  template <typename FromT = From,
            typename ToT   = To,
            typename InputIterator,
            typename OutputIterator,
            typename std::enable_if<std::is_same<FromT, ToT>::value, void>::type* = nullptr>
  void operator()(InputIterator begin, InputIterator end, OutputIterator out) const
  {
    std::copy(begin, end, out);
  }

  // Are the types convertible or can target be constructed from source?
  template <typename FromT = From,
            typename ToT   = To,
            typename InputIterator,
            typename OutputIterator,
            typename std::enable_if<!std::is_same<FromT, ToT>::value &&
                                      (cudf::is_convertible<FromT, ToT>::value ||
                                       std::is_constructible<ToT, FromT>::value),
                                    void>::type* = nullptr>
  void operator()(InputIterator begin, InputIterator end, OutputIterator out) const
  {
    std::transform(begin, end, out, [](auto const& e) { return static_cast<ToT>(e); });
  }

  // Is the target type a fixed_point type
  template <typename FromT = From,
            typename ToT   = To,
            typename InputIterator,
            typename OutputIterator,
            typename std::enable_if<!std::is_same<FromT, ToT>::value &&
                                    cudf::is_fixed_point<ToT>()>::type* = nullptr>
  void operator()(InputIterator begin, InputIterator end, OutputIterator out) const
  {
    std::transform(begin, end, out, [](auto const& e) { return ToT{e, numeric::scale_type{0}}; });
  }

#if 0
  // This is to be used when timestamp disallows construction from tick counts; presently,
  // this conflicts with the convertible/constructible overload
  // Convert integral values to timestamps
  template <
    typename FromT                        = From,
    typename ToT                       = To,
    typename InputIterator, typename OutputIterator,
    typename std::enable_if<std::is_integral<FromT>::value && cudf::is_timestamp_t<ToT>::value,
                            void>::type* = nullptr>
  void operator()(InputIterator begin,
                  InputIterator end,
                  OutputIterator out
                  ) const
  {
      std::transform(
      begin, end, out, [](auto const& e) { return ToT{typename ToT::duration{e}}; });
  }
#endif
};

/**
 * @brief Creates a `device_buffer` containing the elements in the range
 * `[begin,end)`.
 *
 * @tparam ElementTo The type of element that is being created
 * @tparam ElementFrom The type of element that is used to create elements of type `ElementTo`
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @param begin Begining of the sequence of elements
 * @param end End of the sequence of elements
 * @return rmm::device_buffer Buffer containing all elements in the range
 *`[begin,end)`
 **/
template <typename ElementTo, typename ElementFrom, typename InputIterator>
rmm::device_buffer make_elements(InputIterator begin, InputIterator end)
{
  static_assert(cudf::is_fixed_width<ElementTo>(), "Unexpected non-fixed width type.");
  cudf::size_type size = std::distance(begin, end);
  thrust::host_vector<ElementTo> elements;
  elements.reserve(size);
  fixed_width_type_converter<ElementFrom, ElementTo>{}(begin, end, elements.begin());
  return rmm::device_buffer{elements.data(), size * sizeof(ElementTo)};
}

/**
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
 **/
template <typename ValidityIterator>
std::vector<bitmask_type> make_null_mask_vector(ValidityIterator begin, ValidityIterator end)
{
  cudf::size_type size = std::distance(begin, end);
  auto num_words       = cudf::bitmask_allocation_size_bytes(size) / sizeof(bitmask_type);
  std::vector<bitmask_type> null_mask(num_words, 0);
  for (auto i = 0; i < size; ++i) {
    if (*(begin + i)) { set_bit_unsafe(null_mask.data(), i); }
  }
  return null_mask;
}

/**
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
 **/
template <typename ValidityIterator>
rmm::device_buffer make_null_mask(ValidityIterator begin, ValidityIterator end)
{
  auto null_mask = make_null_mask_vector(begin, end);
  return rmm::device_buffer{null_mask.data(),
                            null_mask.size() * sizeof(decltype(null_mask.front()))};
}

/**
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
 **/
template <typename StringsIterator, typename ValidityIterator>
auto make_chars_and_offsets(StringsIterator begin, StringsIterator end, ValidityIterator v)
{
  std::vector<char> chars{};
  std::vector<int32_t> offsets(1, 0);
  for (auto str = begin; str < end; ++str) {
    std::string tmp = (*v++) ? std::string(*str) : std::string{};
    chars.insert(chars.end(), std::cbegin(tmp), std::cend(tmp));
    offsets.push_back(offsets.back() + tmp.length());
  }
  return std::make_pair(std::move(chars), std::move(offsets));
};
}  // namespace detail

/**
 * @brief `column_wrapper` derived class for wrapping columns of fixed-width
 * elements.
 *
 * @tparam ElementTo The fixed-width element type that is created
 * @tparam SourceElementT The fixed-width element type that is used to create elements of type
 * `ElementTo`
 **/
template <typename ElementTo, typename SourceElementT = ElementTo>
class fixed_width_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Default constructor initializes an empty column with proper dtype
   **/
  fixed_width_column_wrapper() : column_wrapper{}
  {
    std::vector<ElementTo> empty;
    wrapped.reset(new cudf::column{
      cudf::data_type{cudf::type_to_id<ElementTo>()},
      0,
      detail::make_elements<ElementTo, SourceElementT>(empty.begin(), empty.end())});
  }

  /**
   * @brief Construct a non-nullable column of the fixed-width elements in the
   * range `[begin,end)`.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable column of INT32 elements with 5 elements: {0, 2, 4, 6, 8}
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i*2;});
   * fixed_width_column_wrapper<int32_t> w(elements, elements + 5);
   * @endcode
   *
   * Note: similar to `std::vector`, this "range" constructor should be used
   *       with parentheses `()` and not braces `{}`. The latter should only
   *       be used for the `initializer_list` constructors
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   **/
  template <typename InputIterator>
  fixed_width_column_wrapper(InputIterator begin, InputIterator end) : column_wrapper{}
  {
    cudf::size_type size = std::distance(begin, end);
    wrapped.reset(new cudf::column{cudf::data_type{cudf::type_to_id<ElementTo>()},
                                   size,
                                   detail::make_elements<ElementTo, SourceElementT>(begin, end)});
  }

  /**
   * @brief Construct a nullable column of the fixed-width elements in the range
   * `[begin,end)` using the range `[v, v + distance(begin,end))` interpreted
   * as booleans to indicate the validity of each element.
   *
   * If `v[i] == true`, element `i` is valid, else it is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable column of INT32 elements with 5 elements: {null, 1, null, 3, null}
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i;});
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;})
   * fixed_width_column_wrapper<int32_t> w(elements, elements + 5, validity);
   * @endcode
   *
   * Note: similar to `std::vector`, this "range" constructor should be used
   *       with parentheses `()` and not braces `{}`. The latter should only
   *       be used for the `initializer_list` constructors
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param v The beginning of the sequence of validity indicators
   **/
  template <typename InputIterator, typename ValidityIterator>
  fixed_width_column_wrapper(InputIterator begin, InputIterator end, ValidityIterator v)
    : column_wrapper{}
  {
    cudf::size_type size = std::distance(begin, end);

    wrapped.reset(new cudf::column{cudf::data_type{cudf::type_to_id<ElementTo>()},
                                   size,
                                   detail::make_elements<ElementTo, SourceElementT>(begin, end),
                                   detail::make_null_mask(v, v + size),
                                   cudf::UNKNOWN_NULL_COUNT});
  }

  /**
   * @brief Construct a non-nullable column of fixed-width elements from an
   * initializer list.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable INT32 column with 4 elements: {1, 2, 3, 4}
   * fixed_width_column_wrapper<int32_t> w{{1, 2, 3, 4}};
   * @endcode
   *
   * @param element_list The list of elements
   **/
  template <typename ElementFrom>
  fixed_width_column_wrapper(std::initializer_list<ElementFrom> elements)
    : fixed_width_column_wrapper(std::cbegin(elements), std::cend(elements))
  {
  }

  /**
   * @brief Construct a nullable column from a list of fixed-width elements
   * using another list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of
   * booleans where `true` indicates the element is valid, and `false` indicates
   * the element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable INT32 column with 4 elements: {1, NULL, 3, NULL}
   * fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, {1, 0, 1, 0}};
   * @endcode
   *
   * @param elements The list of elements
   * @param validity The list of validity indicator booleans
   **/
  template <typename ElementFrom>
  fixed_width_column_wrapper(std::initializer_list<ElementFrom> elements,
                             std::initializer_list<bool> validity)
    : fixed_width_column_wrapper(std::cbegin(elements), std::cend(elements), std::cbegin(validity))
  {
  }

  /**
   * @brief Construct a nullable column from a list of fixed-width elements and
   * the the range `[v, v + element_list.size())` interpreted as booleans to
   * indicate the validity of each element.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable INT32 column with 4 elements: {NULL, 1, NULL, 3}
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;})
   * fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, validity}
   * @endcode
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   * convertible to `bool`
   * @param element_list The list of elements
   * @param v The beginning of the sequence of validity indicators
   **/
  template <typename ValidityIterator, typename ElementFrom>
  fixed_width_column_wrapper(std::initializer_list<ElementFrom> element_list, ValidityIterator v)
    : fixed_width_column_wrapper(std::cbegin(element_list), std::cend(element_list), v)
  {
  }

  /**
   * @brief Construct a nullable column of the fixed-width elements in the range
   * `[begin,end)` using a validity initializer list to indicate the validity of each element.
   *
   * The validity of each element is determined by an `initializer_list` of
   * booleans where `true` indicates the element is valid, and `false` indicates
   * the element is null.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable column of INT32 elements with 5 elements: {null, 1, null, 3, null}
   * fixed_width_column_wrapper<int32_t> w(elements, elements + 5, {0, 1, 0, 1, 0});
   * @endcode
   *
   * @param begin The beginning of the sequence of elements
   * @param end The end of the sequence of elements
   * @param validity The list of validity indicator booleans
   **/
  template <typename InputIterator>
  fixed_width_column_wrapper(InputIterator begin,
                             InputIterator end,
                             std::initializer_list<bool> const& validity)
    : fixed_width_column_wrapper(begin, end, std::cbegin(validity))
  {
  }
};

/**
 * @brief `column_wrapper` derived class for wrapping columns of strings.
 **/
class strings_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Construct a non-nullable column of strings from the range
   * `[begin,end)`.
   *
   * Values in the sequence `[begin,end)` will each be converted to
   *`std::string` and a column will be created containing all of the strings.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable STRING column with 7 string elements:
   * // {"", "this", "is", "a", "column", "of", "strings"}
   * std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
   * strings_column_wrapper s(strings.begin(), strings.end());
   * @endcode
   *
   * @tparam StringsIterator A `std::string` must be constructible from
   * dereferencing a `StringsIterator`.
   * @param begin The beginning of the sequence
   * @param end The end of the sequence
   **/
  template <typename StringsIterator>
  strings_column_wrapper(StringsIterator begin, StringsIterator end) : column_wrapper{}
  {
    std::vector<char> chars;
    std::vector<int32_t> offsets;
    auto all_valid           = make_counting_transform_iterator(0, [](auto i) { return true; });
    std::tie(chars, offsets) = detail::make_chars_and_offsets(begin, end, all_valid);
    wrapped                  = cudf::make_strings_column(chars, offsets);
  }

  /**
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
   * @code{.cpp}
   * // Creates a nullable STRING column with 7 string elements:
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * strings_column_wrapper s(strings.begin(), strings.end(), validity);
   * @endcode
   *
   * @tparam StringsIterator A `std::string` must be constructible from
   * dereferencing a `StringsIterator`.
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   * convertible to `bool`
   * @param begin The beginning of the sequence
   * @param end The end of the sequence
   * @param v The beginning of the sequence of validity indicators
   **/
  template <typename StringsIterator, typename ValidityIterator>
  strings_column_wrapper(StringsIterator begin, StringsIterator end, ValidityIterator v)
    : column_wrapper{}
  {
    size_type num_strings = std::distance(begin, end);
    std::vector<char> chars;
    std::vector<int32_t> offsets;
    std::tie(chars, offsets) = detail::make_chars_and_offsets(begin, end, v);
    wrapped =
      cudf::make_strings_column(chars, offsets, detail::make_null_mask_vector(v, v + num_strings));
  }

  /**
   * @brief Construct a non-nullable column of strings from a list of strings.
   *
   * Example:
   * @code{.cpp}
   * // Creates a non-nullable STRING column with 7 string elements:
   * // {"", "this", "is", "a", "column", "of", "strings"}
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"});
   * @endcode
   *
   * @param strings The list of strings
   **/
  strings_column_wrapper(std::initializer_list<std::string> strings)
    : strings_column_wrapper(std::cbegin(strings), std::cend(strings))
  {
  }

  /**
   * @brief Construct a nullable column of strings from a list of strings and
   * the range `[v, v + strings.size())` interpreted as booleans to indicate the
   * validity of each string.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable STRING column with 7 string elements:
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"}, validity);
   * @endcode
   *
   * @tparam ValidityIterator Dereferencing a ValidityIterator must be
   * convertible to `bool`
   * @param strings The list of strings
   * @param v The beginning of the sequence of validity indicators
   **/
  template <typename ValidityIterator>
  strings_column_wrapper(std::initializer_list<std::string> strings, ValidityIterator v)
    : strings_column_wrapper(std::cbegin(strings), std::cend(strings), v)
  {
  }

  /**
   * @brief Construct a nullable column of strings from a list of strings and
   * a list of booleans to indicate the validity of each string.
   *
   * Example:
   * @code{.cpp}
   * // Creates a nullable STRING column with 7 string elements:
   * // {NULL, "this", NULL, "a", NULL, "of", NULL}
   * strings_column_wrapper s({"", "this", "is", "a", "column", "of", "strings"},
   *                          {0,1,0,1,0,1,0});
   * @endcode
   *
   * @param strings The list of strings
   * @param validity The list of validity indicator booleans
   **/
  strings_column_wrapper(std::initializer_list<std::string> strings,
                         std::initializer_list<bool> validity)
    : strings_column_wrapper(std::cbegin(strings), std::cend(strings), std::cbegin(validity))
  {
  }
};

/**
 * @brief `column_wrapper` derived class for wrapping columns of lists.
 *
 * Important note : due to the way initializer lists work, there is a
 * non-obvious behavioral difference when declaring nested empty lists
 * in different situations.  Specifically,
 *
 * - When compiled inside of a templated class function (such as a TYPED_TEST
 *   cudf test wrapper), nested empty lists behave as they read, semantically.
 *
 * @code{.pseudo}
 *   lists_column_wrapper<int> col{ {LCW{}} }
 *   This yields a List<List<int>> column containing 1 row : a list
 *   containing an empty list.
 * @endcode
 *
 * - When compiled under other situations (a global function, or a non
 *   templated class function), the behavior is different.
 *
 * @code{.pseudo}
 *   lists_column_wrapper<int> col{ {LCW{}} }
 *   This yields a List<int> column containing 1 row that is an empty
 *   list.
 * @endcode
 *
 * This only effects the initial nesting of the empty list. In summary, the
 * correct way to declare an "Empty List" in the two cases are:
 *
 * @code{.pseudo}
 *   // situation 1 (cudf TYPED_TEST case)
 *   LCW{}
 *   // situation 2 (cudf TEST_F case)
 *   {LCW{}}
 * @endcode
 *
 */
template <typename T>
class lists_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Construct a lists column containing a single list of fixed-width
   * type from an initializer list of values.
   *
   * Example:
   * @code{.cpp}
   * Creates a LIST column with 1 list composed of 2 total integers
   * [{0, 1}]
   * lists_column_wrapper l{0, 1};
   * @endcode
   *
   * @param elements The list of elements
   */
  template <typename Element = T, std::enable_if_t<cudf::is_fixed_width<Element>()>* = nullptr>
  lists_column_wrapper(std::initializer_list<T> elements) : column_wrapper{}
  {
    build_from_non_nested(std::move(cudf::test::fixed_width_column_wrapper<T>(elements).release()));
  }

  /**
   * @brief  Construct a lists column containing a single list of fixed-width
   * type from an interator range.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 list composed of 5 total integers
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i*2;});
   * // [{0, 1, 2, 3, 4}]
   * lists_column_wrapper l(elements, elements+5);
   * @endcode
   *
   * @param begin Beginning of the sequence
   * @param end End of the sequence
   */
  template <typename Element = T,
            typename InputIterator,
            std::enable_if_t<cudf::is_fixed_width<Element>()>* = nullptr>
  lists_column_wrapper(InputIterator begin, InputIterator end) : column_wrapper{}
  {
    build_from_non_nested(std::move(
      cudf::test::fixed_width_column_wrapper<typename InputIterator::value_type>(begin, end)
        .release()));
  }

  /**
   * @brief Construct a lists column containing a single list of fixed-width
   * type from an initializer list of values and a validity iterator.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 lists composed of 2 total integers
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [{0, NULL}]
   * lists_column_wrapper l{{0, 1}, validity};
   * @endcode
   *
   * @param elements The list of elements
   * @param v The validity iterator
   */
  template <typename Element = T,
            typename ValidityIterator,
            std::enable_if_t<cudf::is_fixed_width<Element>()>* = nullptr>
  lists_column_wrapper(std::initializer_list<T> elements, ValidityIterator v) : column_wrapper{}
  {
    build_from_non_nested(
      std::move(cudf::test::fixed_width_column_wrapper<T>(elements, v).release()));
  }

  /**
   * @brief Construct a lists column containing a single list of fixed-width
   * type from an iterator range and a validity iterator.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 lists composed of 5 total integers
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i*2;});
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [{0, NULL, 2, NULL, 4}]
   * lists_column_wrapper l(elements, elements+5, validity);
   * @endcode
   *
   * @param begin Beginning of the sequence
   * @param end End of the sequence
   * @param v The validity iterator
   */
  template <typename Element = T,
            typename InputIterator,
            typename ValidityIterator,
            std::enable_if_t<cudf::is_fixed_width<Element>()>* = nullptr>
  lists_column_wrapper(InputIterator begin, InputIterator end, ValidityIterator v)
    : column_wrapper{}
  {
    build_from_non_nested(
      std::move(cudf::test::fixed_width_column_wrapper<T>(begin, end, v).release()));
  }

  /**
   * @brief Construct a lists column containing a single list of strings
   * from an initializer list of values.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 list composed of 2 total strings
   * // [{"abc", "def"}]
   * lists_column_wrapper l{"abc", "def"};
   * @endcode
   *
   * @param elements The list of elements
   */
  template <typename Element                                                   = T,
            std::enable_if_t<std::is_same<Element, cudf::string_view>::value>* = nullptr>
  lists_column_wrapper(std::initializer_list<std::string> elements) : column_wrapper{}
  {
    build_from_non_nested(
      std::move(cudf::test::strings_column_wrapper(elements.begin(), elements.end()).release()));
  }

  /**
   * @brief Construct a lists column containing a single list of strings
   * from an initializer list of values and a validity iterator.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 list composed of 2 total strings
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [{"abc", NULL}]
   * lists_column_wrapper l{{"abc", "def"}, validity};
   * @endcode
   *
   * @param elements The list of elements
   * @param v The validity iterator
   */
  template <typename Element = T,
            typename ValidityIterator,
            std::enable_if_t<std::is_same<Element, cudf::string_view>::value>* = nullptr>
  lists_column_wrapper(std::initializer_list<std::string> elements, ValidityIterator v)
    : column_wrapper{}
  {
    build_from_non_nested(
      std::move(cudf::test::strings_column_wrapper(elements.begin(), elements.end(), v).release()));
  }

  /**
   * @brief Construct a lists column of nested lists from an initializer list of values.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 3 lists
   * // [{0, 1}, {2, 3}, {4, 5}]
   * lists_column_wrapper l{ {0, 1}, {2, 3}, {4, 5} };
   * @endcode
   *
   * Automatically handles nesting
   * Example:
   * @code{.cpp}
   * // Creates a LIST of LIST columns with 2 lists on the top level and
   * // 4 below
   * // [ {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}} ]
   * lists_column_wrapper l{ {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}} };
   * @endcode
   *
   * @param elements The list of elements
   */
  lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> elements) : column_wrapper{}
  {
    std::vector<bool> valids;
    build_from_nested(elements, valids);
  }

  /**
   * @brief Construct am empty lists column
   *
   * Example:
   * @code{.cpp}
   * // Creates an empty LIST column
   * // []
   * lists_column_wrapper l{};
   * @endcode
   *
   */
  lists_column_wrapper() : column_wrapper{}
  {
    build_from_non_nested(make_empty_column(cudf::data_type{cudf::type_to_id<T>()}));
  }

  /**
   * @brief Construct a lists column of nested lists from an initializer list of values
   * and a validity iterator.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 3 lists
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [{0, 1}, NULL, {4, 5}]
   * lists_column_wrapper l{ {{0, 1}, {2, 3}, {4, 5}, validity} };
   * @endcode
   *
   * Automatically handles nesting
   * Example:
   * @code{.cpp}
   * // Creates a LIST of LIST columns with 2 lists on the top level and
   * // 4 below
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [ {{0, 1}, NULL}, {{4, 5}, NULL} ]
   * lists_column_wrapper l{ {{{0, 1}, {2, 3}}, validity}, {{{4, 5}, {6, 7}}, validity} };
   * @endcode
   *
   * @param elements The list of elements
   * @param v The validity iterator
   */
  template <typename ValidityIterator>
  lists_column_wrapper(std::initializer_list<lists_column_wrapper<T>> elements, ValidityIterator v)
    : column_wrapper{}
  {
    std::vector<bool> validity;
    std::transform(elements.begin(),
                   elements.end(),
                   v,
                   std::back_inserter(validity),
                   [](lists_column_wrapper const& l, bool valid) { return valid; });
    build_from_nested(elements, validity);
  }

 private:
  /**
   * @brief Initialize as a nested list column composed of other list columns.
   *
   * This function handles a special case.  For convenience of declaration, we want to treat these
   * two cases as equivalent
   *
   * List<int>      = { 0, 1 }
   * List<int>      = { {0, 1} }
   *
   * while at the same time, allowing further nesting
   * List<List<int> = { {{0, 1}} }
   *
   * @param c Input column to be wrapped
   *
   */
  void build_from_nested(std::initializer_list<lists_column_wrapper<T>> elements,
                         std::vector<bool> const& v)
  {
    auto valids = cudf::test::make_counting_transform_iterator(
      0, [&v](auto i) { return v.empty() ? true : v[i]; });

    // preprocess the incoming lists. unwrap any "root" lists and just use their
    // underlying non-list data.
    // also, sanity check everything to make sure the types of all the columns are the same
    std::vector<column_view> cols;
    type_id child_id = type_id::EMPTY;
    std::transform(elements.begin(),
                   elements.end(),
                   std::back_inserter(cols),
                   [&child_id](lists_column_wrapper const& l) {
                     // potentially unwrap
                     cudf::column_view col =
                       l.root ? lists_column_view(*l.wrapped).child() : *l.wrapped;

                     if (child_id == type_id::EMPTY) {
                       child_id = col.type().id();
                     } else {
                       // handle the case where we have incomplete hierarchies. see comment
                       // below in the concatenate step.
                       if (child_id != type_id::LIST && col.type().id() == type_id::LIST) {
                         child_id = type_id::LIST;
                       }
                     }
                     return col;
                   });

    // generate offsets column and do some type checking to make sure the user hasn't passed an
    // invalid initializer list
    size_type count = 0;
    std::vector<size_type> offsetv;
    std::transform(cols.begin(),
                   cols.end(),
                   valids,
                   std::back_inserter(offsetv),
                   [&](cudf::column_view const& col, bool valid) {
                     // nulls are represented as a repeated offset
                     size_type ret = count;
                     if (valid) { count += col.size(); }
                     return ret;
                   });
    // add the final offset
    offsetv.push_back(count);
    auto offsets =
      cudf::test::fixed_width_column_wrapper<size_type>(offsetv.begin(), offsetv.end()).release();

    // concatenate them together, skipping data for children that are null.
    // handle the corner case where we have an "incomplete hierarchy"
    //
    // lists_column_wrapper<int> list{ {{}}, {} };
    //
    // The above case makes sense semantically. The second list is simply empty and has no internal
    // list. However, because of the way the constructors are called here,  the list on the left
    // will be of type List<List<int>> while the one on the right will be of type List<int> which
    // would normally be appear to be a mismatch of types (as in the case of List<float> and
    // List<int>).  we will allow this specific case through so that we get useful semantics when
    // declaring these objects.
    std::vector<column_view> children;
    for (size_t idx = 0; idx < cols.size(); idx++) {
      if (!valids[idx]) { continue; }

      // handle incomplete hierarchies
      bool is_empty_mismatched_child = (cols[idx].type().id() != child_id) && cols[idx].size() == 0;
      if (is_empty_mismatched_child) { continue; }

      // otherwise we expect that all columns have the same type
      CUDF_EXPECTS(cols[idx].type().id() == child_id, "Unexpected child column mismatch");

      children.push_back(cols[idx]);
    }
    auto data =
      children.empty() ? make_empty_column(cudf::data_type{child_id}) : concatenate(children);

    // construct the list column
    wrapped = make_lists_column(
      cols.size(),
      std::move(offsets),
      std::move(data),
      v.size() <= 0 ? 0 : cudf::UNKNOWN_NULL_COUNT,
      v.size() <= 0 ? rmm::device_buffer{0} : detail::make_null_mask(v.begin(), v.end()));
  }

  /**
   * @brief Initialize as a "root" list column from a non-list input column.  Root columns
   * will be "unwrapped" when used in the nesting (list of lists) case.
   *
   * @param c Input column to be wrapped
   *
   */
  void build_from_non_nested(std::unique_ptr<column> c)
  {
    CUDF_EXPECTS(c->type().id() == type_id::EMPTY || !cudf::is_nested(c->type()),
                 "Unexpected type");

    std::vector<size_type> offsetv;
    offsetv.push_back(0);
    if (c->size() > 0) { offsetv.push_back(c->size()); }
    auto offsets =
      cudf::test::fixed_width_column_wrapper<size_type>(offsetv.begin(), offsetv.end()).release();

    // construct the list column. mark this as a root
    root    = true;
    wrapped = make_lists_column(
      offsetv.size() - 1, std::move(offsets), std::move(c), 0, rmm::device_buffer{0});
  }

  bool root = false;
};
}  // namespace test
}  // namespace cudf
