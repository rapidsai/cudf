/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/copy.h>

#include <algorithm>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace test {

// Forward declaration for lists_column_initializer
template <typename T, typename SourceElementT>
class lists_column_wrapper;

template <typename Iterator>
concept iterator_like = requires(Iterator i) {
  *i;
  ++i;
};

template <typename Iterator>
concept validity_iterator =
  iterator_like<Iterator> && requires(Iterator i) { static_cast<bool>(*i); };

/**
 * @brief Host-side recursive initializer tree for constructing list columns with an
 * explicit stream and memory resources at every nesting level.
 *
 * Example:
 * @code{.cpp}
 * using LCW = cudf::test::lists_column_wrapper<int>;
 * LCW col{{{1, 2}, {3}}, stream, mr};
 * @endcode
 *
 * Leaf and nested constructors accept the existing validity iterators
 * (`valids`, `null_at(...)`, etc.) and materialize them into owned storage.
 *
 * @tparam T Host leaf element type (e.g. `int32_t` or `std::string`)
 */
template <typename T>
class lists_column_initializer {
 public:
  using value_type = T;

  struct string_element {
    string_element(char const* value) : value_{value} {}
    string_element(std::string value) : value_{std::move(value)} {}

    std::string value_;
  };
  /**
   * @brief Construct an empty leaf. Avoids ambiguity between the leaf and nested
   * empty `initializer_list` constructors.
   */
  lists_column_initializer() = default;

  /**
   * @brief Construct a leaf from scalar values.
   *
   * @param values Leaf element values
   */
  template <typename Element>
  lists_column_initializer(std::initializer_list<Element> values)
    requires(!std::is_same_v<T, std::string> && std::is_convertible_v<Element, T>)
    : values_(values.begin(), values.end())
  {
  }

  template <typename First, typename... Rest>
  lists_column_initializer(First first, Rest... rest)
    requires(sizeof...(Rest) > 0 && !std::is_same_v<T, std::string> &&
             std::is_convertible_v<First, T> && (std::is_convertible_v<Rest, T> && ...))
    : values_{static_cast<T>(first), static_cast<T>(rest)...}
  {
  }

  lists_column_initializer(std::initializer_list<string_element> values)
    requires(std::is_same_v<T, std::string>)
  {
    values_.reserve(values.size());
    std::transform(
      values.begin(), values.end(), std::back_inserter(values_), [](auto const& value) {
        return value.value_;
      });
  }

  template <iterator_like InputIterator>
  lists_column_initializer(InputIterator begin, InputIterator end) : values_(begin, end)
  {
  }

  /**
   * @brief Construct a leaf from scalar values and a validity iterator.
   *
   * @tparam ValidityIterator Iterator convertible to `bool`
   * @param values Leaf element values
   * @param v Validity iterator over `values.size()` elements
   */
  template <validity_iterator ValidityIterator>
  lists_column_initializer(std::initializer_list<T> values, ValidityIterator v)
    requires(!std::is_same_v<T, std::string>)
    : values_{values}, has_validity_{true}
  {
    value_validity_.reserve(values_.size());
    for (std::size_t i = 0; i < values_.size(); ++i) {
      value_validity_.push_back(static_cast<bool>(*v++));
    }
  }

  template <validity_iterator ValidityIterator>
  lists_column_initializer(std::initializer_list<string_element> values, ValidityIterator v)
    requires(std::is_same_v<T, std::string>)
    : has_validity_{true}
  {
    values_.reserve(values.size());
    value_validity_.reserve(values.size());
    for (auto const& value : values) {
      values_.push_back(value.value_);
      value_validity_.push_back(static_cast<bool>(*v++));
    }
  }

  lists_column_initializer(std::initializer_list<T> values, std::initializer_list<bool> validity)
    requires(!std::is_same_v<T, std::string>)
    : lists_column_initializer(values, validity.begin())
  {
  }

  lists_column_initializer(std::initializer_list<string_element> values,
                           std::initializer_list<bool> validity)
    requires(std::is_same_v<T, std::string>)
    : lists_column_initializer(values, validity.begin())
  {
  }

  /**
   * @brief Construct a nested node from child initializers.
   *
   * This constructor is a template so the non-template leaf
   * `initializer_list<T>` constructor is preferred for scalar lists such as
   * `{1, 2, 3}`. Otherwise both overloads are non-templates and constructing
   * `Init` from an `int` via the nested overload recurses until the stack
   * overflows.
   *
   * @param children Child list initializers
   */
  template <typename NestedInit = lists_column_initializer>
  lists_column_initializer(std::initializer_list<NestedInit> children)
    requires(std::is_same_v<NestedInit, lists_column_initializer>)
    : children_{children.begin(), children.end()}, nested_{true}
  {
  }

  static lists_column_initializer nested(std::initializer_list<lists_column_initializer> children)
  {
    return lists_column_initializer(children);
  }

  template <validity_iterator ValidityIterator>
  static lists_column_initializer nested(std::initializer_list<lists_column_initializer> children,
                                         ValidityIterator v)
  {
    return lists_column_initializer(children, v);
  }

  /**
   * @brief Construct a nested node from child initializers and a row-validity iterator.
   *
   * @tparam ValidityIterator Iterator convertible to `bool`
   * @param children Child list initializers
   * @param v Validity iterator over `children.size()` rows
   */
  template <validity_iterator ValidityIterator, typename NestedInit = lists_column_initializer>
  lists_column_initializer(std::initializer_list<NestedInit> children, ValidityIterator v)
    requires(std::is_same_v<NestedInit, lists_column_initializer>)
    : nested_{true}, has_validity_{true}
  {
    children_.reserve(children.size());
    for (auto const& child : children) {
      children_.push_back(child);
      children_.back().valid_ = static_cast<bool>(*v++);
    }
  }

  template <typename NestedInit = lists_column_initializer>
  lists_column_initializer(std::initializer_list<NestedInit> children,
                           std::initializer_list<bool> validity)
    requires(std::is_same_v<NestedInit, lists_column_initializer>)
    : lists_column_initializer(children, validity.begin())
  {
  }

  /**
   * @brief True if this node holds nested child initializers rather than leaf values.
   * @return Whether this node is nested
   */
  [[nodiscard]] bool nested() const { return nested_; }
  /**
   * @brief True if this row is valid (non-null) in its parent list.
   * @return Whether this row is valid
   */
  [[nodiscard]] bool valid() const { return valid_; }
  /**
   * @brief Leaf element values when `nested()` is false.
   * @return Reference to the leaf values
   */
  [[nodiscard]] auto const& values() const { return values_; }
  /**
   * @brief Per-element validity for leaf values; empty when all leaf values are valid.
   * @return Reference to the leaf validity mask
   */
  [[nodiscard]] auto const& value_validity() const { return value_validity_; }
  /**
   * @brief True if validity was explicitly provided.
   * @return Whether this initializer has explicit validity
   */
  [[nodiscard]] bool has_validity() const { return has_validity_; }
  /**
   * @brief Child initializers when `nested()` is true.
   * @return Reference to the child initializers
   */
  [[nodiscard]] auto const& children() const { return children_; }

  /**
   * @brief Recursively build child list wrappers and row validity for a nested node.
   *
   * Each valid child is allocated with the provided `stream` and `mr`. Null children are
   * represented as default-constructed wrappers (skipped during concatenate).
   *
   * @tparam ElementT List wrapper element type
   * @tparam SourceElementT Source type used by the list wrapper
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate child columns
   * @return Child wrappers and an empty validity vector when all rows are valid,
   *         otherwise a validity mask matching `children().size()`
   */
  template <typename ElementT, typename SourceElementT = ElementT>
  [[nodiscard]] std::pair<std::vector<lists_column_wrapper<ElementT, SourceElementT>>,
                          std::vector<bool>>
  build(cuda::stream_ref stream, cudf::memory_resources mr) &&
  {
    std::vector<lists_column_wrapper<ElementT, SourceElementT>> children;
    std::vector<bool> validity;
    children.reserve(children_.size());
    validity.reserve(children_.size());
    for (auto&& child : children_) {
      validity.push_back(child.valid());
      children.emplace_back(std::move(child), stream, mr);
    }
    return {std::move(children), has_validity_ ? std::move(validity) : std::vector<bool>{}};
  }

 private:
  template <validity_iterator ValidityIterator>
  lists_column_initializer with_validity(ValidityIterator validity) &&
  {
    has_validity_ = true;
    if (nested_) {
      for (auto& child : children_) {
        child.valid_ = static_cast<bool>(*validity++);
      }
    } else {
      value_validity_.clear();
      value_validity_.reserve(values_.size());
      std::transform(validity,
                     validity + values_.size(),
                     std::back_inserter(value_validity_),
                     [](auto const& value) { return static_cast<bool>(value); });
    }
    return std::move(*this);
  }

  template <typename, typename>
  friend class lists_column_wrapper;

  std::vector<T> values_;
  std::vector<bool> value_validity_;
  std::vector<lists_column_initializer> children_;
  bool nested_{false};
  bool valid_{true};
  bool has_validity_{false};
};

/**
 * @brief `column_wrapper` derived class for wrapping columns of lists.
 *
 * Nested rows can normally be expressed directly with braces. Use `nested()` when a
 * single-child initializer would otherwise make the intended depth ambiguous.
 *
 * @code{.cpp}
 * using LCW = cudf::test::lists_column_wrapper<int>;
 * LCW lists{{{1}, {2}}, {}, {{3}, {4, 5}}};
 * LCW deeper{LCW::nested({{1}}), {}};
 * @endcode
 */
template <typename T, typename SourceElementT = T>
class lists_column_wrapper : public detail::column_wrapper {
 public:
  /**
   * @brief Cast to lists_column_view
   */
  operator lists_column_view() const { return cudf::lists_column_view{wrapped->view()}; }

  /**
   * @brief Host-side leaf element type (`std::string` for string lists, else `SourceElementT`).
   */
  using host_element_t =
    std::conditional_t<std::is_same_v<T, cudf::string_view>, std::string, SourceElementT>;
  using initializer_type = lists_column_initializer<host_element_t>;

  static initializer_type nested(std::initializer_list<initializer_type> children)
  {
    return initializer_type::nested(children);
  }

  template <validity_iterator ValidityIterator>
  static initializer_type nested(std::initializer_list<initializer_type> children,
                                 ValidityIterator validity)
  {
    return initializer_type::nested(children, validity);
  }

  /**
   * @brief Column wrapper type used to materialize leaf list contents.
   */
  using leaf_wrapper_t = std::conditional_t<std::is_same_v<T, cudf::string_view>,
                                            strings_column_wrapper,
                                            fixed_width_column_wrapper<T, SourceElementT>>;

  /**
   * @brief Construct a lists column containing a single list from an initializer
   * list of values.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 list composed of 2 total integers
   * // [{0, 1}]
   * lists_column_wrapper l{0, 1};
   * @endcode
   *
   * These leaf constructors are templates (via `requires`) so that recursive initializer
   * construction is preferred for ambiguous cases such as
   * `lists_column_wrapper<cudf::string_view>{{}, {}}`.
   *
   * @param elements The list of elements
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename Element>
  lists_column_wrapper(std::initializer_list<Element> elements,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    requires(cudf::is_fixed_width<T>() && std::is_convertible_v<Element, SourceElementT>)
    : column_wrapper{}
  {
    build_from_non_nested(
      fixed_width_column_wrapper<T, SourceElementT>(elements.begin(), elements.end(), stream, mr)
        .release(),
      stream,
      mr);
  }

  template <typename First, typename... Rest>
  lists_column_wrapper(First first, Rest... rest)
    requires(cudf::is_fixed_width<T>() && (std::is_convertible_v<First, SourceElementT> && ... &&
                                           std::is_convertible_v<Rest, SourceElementT>))
    : lists_column_wrapper(std::initializer_list<SourceElementT>{
        static_cast<SourceElementT>(first), static_cast<SourceElementT>(rest)...})
  {
  }

  /**
   * @brief Construct a lists column containing a single list from an iterator range.
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
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename InputIterator>
    requires requires(InputIterator i) {
      *i;
      ++i;
    }
  lists_column_wrapper(InputIterator begin,
                       InputIterator end,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    build_from_non_nested(leaf_wrapper_t(begin, end, stream, mr).release(), stream, mr);
  }

  /**
   * @brief Construct a lists column containing a single list from an initializer
   * list of values and a validity iterator.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 list composed of 2 total integers
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [{0, NULL}]
   * lists_column_wrapper l{{0, 1}, validity};
   * @endcode
   *
   * @param elements The list of elements
   * @param v The validity iterator
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename Element, typename ValidityIterator>
  lists_column_wrapper(std::initializer_list<Element> elements,
                       ValidityIterator v,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    requires(cudf::is_fixed_width<T>() && std::is_convertible_v<Element, SourceElementT>)
    : column_wrapper{}
  {
    build_from_non_nested(
      fixed_width_column_wrapper<T, SourceElementT>(elements.begin(), elements.end(), v, stream, mr)
        .release(),
      stream,
      mr);
  }

  /**
   * @brief Construct a lists column containing a single list from an iterator
   * range and a validity iterator.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 list composed of 5 total integers
   * auto elements = make_counting_transform_iterator(0, [](auto i){return i*2;});
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [{0, NULL, 2, NULL, 4}]
   * lists_column_wrapper l(elements, elements+5, validity);
   * @endcode
   *
   * @param begin Beginning of the sequence
   * @param end End of the sequence
   * @param v The validity iterator
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename InputIterator, typename ValidityIterator>
    requires requires(InputIterator i) {
      *i;
      ++i;
    }
  lists_column_wrapper(InputIterator begin,
                       InputIterator end,
                       ValidityIterator v,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    build_from_non_nested(leaf_wrapper_t(begin, end, v, stream, mr).release(), stream, mr);
  }

  /**
   * @brief Construct a lists column containing a single list of strings.
   *
   * Example:
   * @code{.cpp}
   * // Creates a LIST column with 1 list composed of 2 total strings
   * // [{"abc", "def"}]
   * lists_column_wrapper<cudf::string_view> s{"abc", "def"};
   * @endcode
   *
   * @param elements The list of strings
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename Element>
  lists_column_wrapper(std::initializer_list<Element> elements,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    requires(std::is_same_v<T, cudf::string_view> && std::is_constructible_v<std::string, Element>)
    : column_wrapper{}
  {
    build_from_non_nested(
      strings_column_wrapper(elements.begin(), elements.end(), stream, mr).release(), stream, mr);
  }

  /**
   * @brief Construct a lists column containing a single list of strings and a
   * validity iterator.
   *
   * Example:
   * @code{.cpp}
   * auto validity = make_counting_transform_iterator(0, [](auto i){return i%2;});
   * // [{"abc", NULL}]
   * lists_column_wrapper<cudf::string_view> l{{"abc", "def"}, validity};
   * @endcode
   *
   * @param elements The list of strings
   * @param v The validity iterator
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename Element, typename ValidityIterator>
  lists_column_wrapper(std::initializer_list<Element> elements,
                       ValidityIterator v,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    requires(std::is_same_v<T, cudf::string_view> && std::is_constructible_v<std::string, Element>)
    : column_wrapper{}
  {
    build_from_non_nested(
      strings_column_wrapper(elements.begin(), elements.end(), v, stream, mr).release(),
      stream,
      mr);
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
   * An explicit stream and memory resource are propagated through every nesting level:
   * `lists_column_wrapper<int> l{{{0, 1}, {2, 3}, {4, 5}}, stream, mr};`
   *
   * @param elements The list of elements
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  lists_column_wrapper(std::initializer_list<initializer_type> elements,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    : lists_column_wrapper(initializer_type(elements), stream, mr)
  {
  }

  /**
   * @brief Construct an empty lists column
   *
   * Example:
   * @code{.cpp}
   * // Creates an empty LIST column
   * // []
   * lists_column_wrapper l{};
   * @endcode
   */
  lists_column_wrapper() : column_wrapper{}
  {
    // Mark as a root so nesting unwraps to the empty child, matching
    // build_from_non_nested on an empty leaf.
    root    = true;
    depth   = 0;
    wrapped = make_empty_lists_column(data_type{type_to_id<T>()});
  }

  /**
   * @brief Construct an empty lists column with an explicit stream and memory resource
   *
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  lists_column_wrapper(cuda::stream_ref stream, cudf::memory_resources mr)
    : lists_column_wrapper(initializer_type{}, stream, mr)
  {
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
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  template <typename ValidityIterator>
  lists_column_wrapper(std::initializer_list<initializer_type> elements,
                       ValidityIterator v,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    : lists_column_wrapper(initializer_type(elements, v), stream, mr)
  {
  }

  /**
   * @brief Construct a lists column from a recursive `lists_column_initializer` tree.
   *
   * Every nesting level is allocated with the provided `stream` and `mr`. Prefer this over
   * brace-nested `lists_column_wrapper` constructions that pass resources only at the outer
   * level.
   *
   * Example:
   * @code{.cpp}
   * using LCW = cudf::test::lists_column_wrapper<int>;
   * LCW lists{{{0, 1}, {2, 3}, {4, 5}}, stream, mr};
   * LCW nested{{LCW::nested({{0, 1}, {2}}), LCW::nested({{3}})}, stream, mr};
   * @endcode
   *
   * @param init Host-side nested values (and optional validity)
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   */
  lists_column_wrapper(initializer_type init,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    : column_wrapper{}
  {
    if (!init.nested()) {
      if (!init.has_validity()) {
        *this = lists_column_wrapper(init.values().begin(), init.values().end(), stream, mr);
      } else {
        *this = lists_column_wrapper(
          init.values().begin(), init.values().end(), init.value_validity().begin(), stream, mr);
      }
      return;
    }

    if (init.children().empty()) {
      wrapped = make_empty_lists_column(data_type{type_to_id<T>()});
      depth   = 1;
      return;
    }

    auto [children, validity] = std::move(init).template build<T, SourceElementT>(stream, mr);
    build_from_nested(children, validity, stream, mr);
  }

  template <validity_iterator ValidityIterator>
  lists_column_wrapper(initializer_type init,
                       ValidityIterator validity,
                       cuda::stream_ref stream   = cudf::test::get_default_stream(),
                       cudf::memory_resources mr = cudf::get_current_device_resource_ref())
    : lists_column_wrapper(std::move(init).with_validity(validity), stream, mr)
  {
  }

  /**
   * @brief Construct a list column containing a single empty, optionally null row.
   *
   * @param valid Whether or not the empty row is also null
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   * @return A list column containing a single empty row
   */
  static lists_column_wrapper make_one_empty_row_column(
    bool valid                = true,
    cuda::stream_ref stream   = cudf::test::get_default_stream(),
    cudf::memory_resources mr = cudf::get_current_device_resource_ref())
  {
    cudf::test::fixed_width_column_wrapper<int32_t> offsets({0, 0}, stream, mr);
    leaf_wrapper_t values(std::initializer_list<host_element_t>{}, stream, mr);
    return lists_column_wrapper(
      1,
      offsets.release(),
      values.release(),
      valid ? 0 : 1,
      valid ? rmm::device_buffer{}
            : cudf::create_null_mask(1, cudf::mask_state::ALL_NULL, stream, mr.get_output_mr()));
  }

 private:
  /**
   * @brief Construct a list column from constituent parts.
   *
   * @param num_rows The number of lists the column represents
   * @param offsets The column of offset values for this column
   * @param values The column of values bounded by the offsets
   * @param null_count The number of null list entries
   * @param null_mask The bits specifying the null lists in device memory
   */
  lists_column_wrapper(size_type num_rows,
                       std::unique_ptr<cudf::column>&& offsets,
                       std::unique_ptr<cudf::column>&& values,
                       size_type null_count,
                       rmm::device_buffer&& null_mask)
  {
    // construct the list column
    wrapped = make_lists_column(
      num_rows, std::move(offsets), std::move(values), null_count, std::move(null_mask));
  }

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
   * @param elements Input columns to be wrapped
   * @param v The validity of each row
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   *
   */
  template <typename ListsRange>
  void build_from_nested(ListsRange const& elements,
                         std::vector<bool> const& v,
                         cuda::stream_ref stream,
                         cudf::memory_resources mr)
  {
    auto valids = cudf::detail::make_counting_transform_iterator(
      0, [&v](auto i) { return v.empty() ? true : v[i]; });

    // compute the expected hierarchy and depth
    auto const hierarchy_and_depth =
      std::accumulate(elements.begin(),
                      elements.end(),
                      std::pair<column_view, int32_t>{{}, -1},
                      [](auto acc, lists_column_wrapper const& lcw) {
                        return lcw.depth > acc.second ? std::pair(lcw.get_view(), lcw.depth) : acc;
                      });
    column_view expected_hierarchy = hierarchy_and_depth.first;
    int32_t const expected_depth   = hierarchy_and_depth.second;

    // preprocess columns so that every column_view in 'cols' is an equivalent hierarchy
    auto [cols, stubs] = preprocess_columns(
      elements, expected_hierarchy, expected_depth, stream, mr.get_temporary_mr());

    // generate offsets
    size_type count = 0;
    std::vector<size_type> offsetv;
    std::transform(cols.cbegin(),
                   cols.cend(),
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
      cudf::test::fixed_width_column_wrapper<int32_t>(offsetv.begin(), offsetv.end(), stream, mr)
        .release();

    // concatenate them together, skipping children that are null.
    std::vector<column_view> children;
    thrust::copy_if(std::cbegin(cols),
                    std::cend(cols),
                    valids,
                    std::back_inserter(children),
                    cuda::std::identity{});

    auto data = children.empty() ? cudf::empty_like(expected_hierarchy)
                                 : cudf::concatenate(children, stream, mr.get_output_mr());

    // increment depth
    depth = expected_depth + 1;

    auto [null_mask, null_count] = [&] {
      if (v.size() <= 0) return std::make_pair(rmm::device_buffer{}, cudf::size_type{0});
      return cudf::test::detail::make_null_mask(v.begin(), v.end(), stream, mr);
    }();

    // construct the list column
    wrapped = make_lists_column(
      cols.size(), std::move(offsets), std::move(data), null_count, std::move(null_mask));
  }

  /**
   * @brief Initialize as a "root" list column from a non-list input column.  Root columns
   * will be "unwrapped" when used in the nesting (list of lists) case.
   *
   * @param c Input column to be wrapped
   * @param stream CUDA stream used for device memory operations
   * @param mr Memory resources used to allocate the returned column
   *
   */
  void build_from_non_nested(std::unique_ptr<column> c,
                             cuda::stream_ref stream,
                             cudf::memory_resources mr)
  {
    CUDF_EXPECTS(c->type().id() == type_id::EMPTY || !cudf::is_nested(c->type()),
                 "Unexpected type");

    std::vector<size_type> offsetv;
    if (c->size() > 0) {
      offsetv.push_back(0);
      offsetv.push_back(c->size());
    }
    auto offsets =
      cudf::test::fixed_width_column_wrapper<int32_t>(offsetv.begin(), offsetv.end(), stream, mr)
        .release();

    // construct the list column. mark this as a root
    root  = true;
    depth = 0;

    size_type num_elements = offsets->size() == 0 ? 0 : offsets->size() - 1;
    wrapped =
      make_lists_column(num_elements, std::move(offsets), std::move(c), 0, rmm::device_buffer{});
  }

  /**
   * @brief Given an input column that may be an "incomplete hierarchy" due to being empty
   * at a level before the leaf, normalize it so that it matches the expected hierarchy of
   * sibling columns.
   *
   * cudf functions that handle lists expect that all columns are fully formed hierarchies,
   * even if they are empty somewhere in the middle of the hierarchy.
   * If we had the following lists_column_wrapper<int> declaration:
   *
   * @code{.pseudo}
   * [ {{{1, 2, 3}}}, {} ]
   * Row 0 in this case is a List<List<List<int>>>, where row 1 appears to be just a List<>.
   * @endcode
   *
   * These two columns will end up getting passed to cudf::concatenate() to merge. But
   * concatenate() will throw an exception because row 1 will appear to have a child type
   * of nothing, while row 0 will appear to have a child type of List<List<int>>.
   * To handle this cleanly, we want to "normalize" row 1 so that it appears as a
   * List<List<List<int>>> column even though it has 0 elements at the top level.
   *
   * This function also detects the case where the user has constructed a truly invalid
   * pair of columns, such as
   *
   * @code{.pseudo}
   * [ {{{1, 2, 3}}}, {4, 5} ]
   * Row 0 in this case is a List<List<List<int>>>, and row 1 is a concrete List<int> with
   * elements. This is purely an invalid way of constructing a lists column.
   * @endcode
   *
   * @param col Input column to be normalized
   * @param expected_hierarchy Input column which represents the expected hierarchy
   * @param stream CUDA stream used for device memory operations
   * @param temp_mr Device memory resource used for temporary normalized copies
   *
   * @return A new column representing a normalized copy of col
   */
  std::unique_ptr<column> normalize_column(column_view const& col,
                                           column_view const& expected_hierarchy,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref temp_mr)
  {
    // if are at the bottom of the short column, it must be empty
    if (col.type().id() != type_id::LIST) {
      CUDF_EXPECTS(col.is_empty(), "Encountered mismatched column!");

      auto remainder = empty_like(expected_hierarchy);
      return remainder;
    }

    lists_column_view lcv(col);
    return make_lists_column(col.size(),
                             std::make_unique<column>(lcv.offsets(), stream, temp_mr),
                             normalize_column(lists_column_view(col).child(),
                                              lists_column_view(expected_hierarchy).child(),
                                              stream,
                                              temp_mr),
                             col.null_count(),
                             cudf::copy_bitmask(col, stream, temp_mr));
  }

  template <typename ListsRange>
  std::pair<std::vector<column_view>, std::vector<std::unique_ptr<column>>> preprocess_columns(
    ListsRange const& elements,
    column_view& expected_hierarchy,
    int expected_depth,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref temp_mr)
  {
    std::vector<std::unique_ptr<column>> stubs;
    std::vector<column_view> cols;

    // preprocess the incoming lists.
    // - unwrap any "root" lists
    // - handle incomplete hierarchies
    std::transform(
      elements.begin(),
      elements.end(),
      std::back_inserter(cols),
      [&](lists_column_wrapper const& l) -> column_view {
        // depth mismatch.  attempt to normalize the short column.
        // this function will also catch if this is a legitimately broken
        // set of input
        if (l.depth < expected_depth) {
          if (l.root) {
            // this exception distinguishes between the following two cases:
            //
            // { {{{1, 2, 3}}}, {} }
            // In this case, row 0 is a List<List<List<int>>>, whereas row 1 is
            // just a List<> which is an apparent mismatch.  However, because row 1
            // is empty we will allow that to semantically mean
            // "a List<List<List<int>>> that's empty at the top level"
            //
            // { {{{1, 2, 3}}}, {4, 5, 6} }
            // In this case, row 1 is a concrete List<int> with actual values.
            // There is no way to rectify the differences so we will treat it as a
            // true column mismatch.
            CUDF_EXPECTS(l.wrapped->size() == 0, "Mismatch in column types!");
            stubs.push_back(empty_like(expected_hierarchy));
          } else {
            stubs.push_back(normalize_column(l.get_view(), expected_hierarchy, stream, temp_mr));
          }
          return *(stubs.back());
        }
        // the empty hierarchy case
        return l.get_view();
      });

    return {std::move(cols), std::move(stubs)};
  }

  [[nodiscard]] column_view get_view() const
  {
    return root ? lists_column_view(*wrapped).child() : *wrapped;
  }

  int depth = 0;
  bool root = false;
};

}  // namespace test
}  // namespace CUDF_EXPORT cudf
