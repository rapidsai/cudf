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
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.cuh>
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
 * auto iter = make_counting_transform_iterator(0, [](auto i) { return (i * i);
 *}); iter[0] == 0 iter[1] == 1 iter[2] == 4
 * ...
 * iter[n] == n * n
 * ```
 *
 * @param start The starting value of the counting iterator
 * @param f The unary function to apply to the counting iterator
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
 * This should not be instantiated directly.
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

 protected:
  std::unique_ptr<cudf::column> wrapped{};  ///< The wrapped column
};

/**---------------------------------------------------------------------------*
 * @brief Creates a `device_buffer` containing the elements in the range
 * `[begin,end)`.
 *
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @param begin Begining of the sequence of elements
 * @param end End of the sequence of elements
 * @return rmm::device_buffer Buffer containing all elements in the range
 *`[begin,end)`
 *---------------------------------------------------------------------------**/
template <typename InputIterator>
static rmm::device_buffer make_elements(InputIterator begin,
                                        InputIterator end) {
  using Element = decltype(*begin);
  static_assert(cudf::is_fixed_width<Element>(),
                "Unexpected non-fixed width type.");
  std::vector<Element> elements(begin, end);
  return rmm::device_buffer{elements.data(), elements.size() * sizeof(Element)};
}

/**---------------------------------------------------------------------------*
 * @brief Create a `device_buffer` containing a validity indicator bitmask using
 * the range `[begin,end)` interpretted as booleans to indicate the state of
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
  cudf::size_type size = std::distance(begin, end);
  std::vector<uint8_t> null_mask(cudf::bitmask_allocation_size_bytes(size), 0);

  for (auto i = 0; i < size; ++i) {
    if (begin[i] == true) {
      set_bit_unsafe(reinterpret_cast<cudf::bitmask_type*>(null_mask.data()),
                     i);
    }
  }
  return rmm::device_buffer{null_mask.data(),
                            null_mask.size() * sizeof(uint8_t)};
}
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
   * @brief Construct a nullable column of the fixed-width elements in the range
   * `[begin,end)` using the range `[v, v + distance(begin,end))` interpretted
   * as booleans to indicate the validity of each element.
   *
   * If `v[i] == true`, element `i` is valid, else it is null.
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
        detail::make_elements(begin, end), detail::make_null_mask(v, v + size),
        cudf::UNKNOWN_NULL_COUNT});
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a non-nullable column of the fixed-width elements in the
   * range `[begin,end)`.
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
        detail::make_elements(begin, end)});
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
      : fixed_width_column_wrapper{std::cbegin(elements), std::cend(elements)} {
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
   * fixed_width_column_wrapper<int32_t> w{ {1,2,3,4}, {true, false, true,
   *false}}
   * ```
   *
   * @param elements The list of elements
   * @param validity The list of validity indicator booleans
   *---------------------------------------------------------------------------**/
  fixed_width_column_wrapper(std::initializer_list<Element> elements,
                             std::initializer_list<bool> validity)
      : fixed_width_column_wrapper{std::cbegin(elements), std::cend(elements),
                                   std::cbegin(validity)} {}

  /**---------------------------------------------------------------------------*
   * @brief Construct a nullable column from a list of fixed-width elements and
   * the the range `[v, v + element_list.size())` interpretted as booleans to
   * indicate the validity of each element.
   *
   * @tparam ValidityIterator
   * @param element_list The list of elements
   * @param v The beginning of the sequence of validity indicators
   *---------------------------------------------------------------------------**/
  template <typename ValidityIterator>
  fixed_width_column_wrapper(std::initializer_list<Element> element_list,
                             ValidityIterator v)
      : fixed_width_column_wrapper{std::cbegin(element_list),
                                   std::cend(element_list), v} {}

};

// std::unique_ptr<cudf::column> create_strings_column(
//    const std::vector<const char*>& h_strings) {
//  cudf::size_type memsize = 0;
//  for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr)
//    memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
//  if (memsize == 0 && h_strings.size())
//    memsize = 1;  // prevent vectors from being null in all empty-string
//    case
//  cudf::size_type count = (cudf::size_type)h_strings.size();
//  thrust::host_vector<char> h_buffer(memsize);
//  thrust::device_vector<char> d_buffer(memsize);
//  thrust::host_vector<thrust::pair<const char*, size_type>> strings(count);
//  cudf::size_type offset = 0;
//  for (cudf::size_type idx = 0; idx < count; ++idx) {
//    const char* str = h_strings[idx];
//    if (!str)
//      strings[idx] = thrust::pair<const char*, size_type>{nullptr, 0};
//    else {
//      cudf::size_type length = (cudf::size_type)strlen(str);
//      memcpy(h_buffer.data() + offset, str, length);
//      strings[idx] = thrust::pair<const char*, size_type>{
//          d_buffer.data().get() + offset, length};
//      offset += length;
//    }
//  }
//  rmm::device_vector<thrust::pair<const char*, size_type>>
//  d_strings(strings); cudaMemcpy(d_buffer.data().get(), h_buffer.data(),
//  memsize,
//             cudaMemcpyHostToDevice);
//  return cudf::make_strings_column(d_strings);
//}

class strings_column_wrapper : public detail::column_wrapper {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Construct a new strings column wrapper object
   *
   * @tparam StringsIterator
   * @param begin
   * @param end
   *---------------------------------------------------------------------------**/
  template <typename StringsIterator>
  strings_column_wrapper(StringsIterator begin, StringsIterator end)
      : column_wrapper{} {}

  /**---------------------------------------------------------------------------*
   * @brief Construct a new strings column wrapper object
   *
   * @tparam StringsIterator
   * @tparam ValidityIterator
   * @param begin
   * @param end
   * @param v
   *---------------------------------------------------------------------------**/
  template <typename StringsIterator, typename ValidityIterator>
  strings_column_wrapper(StringsIterator begin, StringsIterator end,
                         ValidityIterator v)
      : column_wrapper{} {}

  /**---------------------------------------------------------------------------*
   * @brief Construct a new strings column wrapper object
   *
   * @tparam ValidIterator
   * @param strings
   * @param v
   *---------------------------------------------------------------------------**/
  template <typename ValidIterator>
  strings_column_wrapper(std::initializer_list<std::string> strings,
                         ValidIterator v)
      : strings_column_wrapper(std::cbegin(strings), std::cend(strings), v) {}

  /**---------------------------------------------------------------------------*
   * @brief Construct a new strings column wrapper object
   *
   * @param strings
   * @param validity
   *---------------------------------------------------------------------------**/
  strings_column_wrapper(std::initializer_list<std::string> strings,
                         std::initializer_list<bool> validity)
      : strings_column_wrapper(std::cbegin(strings), std::cend(strings),
                               std::cbegin(validity)) {}
};

}  // namespace test
}  // namespace cudf