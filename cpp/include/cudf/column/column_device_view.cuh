/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/detail/utilities/alignment.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/optional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>

#include <algorithm>

/**
 * @file column_device_view.cuh
 * @brief Column device view class definitions
 */

namespace cudf {

/**
 * @brief Indicates the presence of nulls at compile-time or runtime.
 *
 * If used at compile-time, this indicator can tell the optimizer
 * to include or exclude any null-checking clauses.
 *
 * @ingroup column_classes
 *
 */
struct nullate {
  struct YES : std::bool_constant<true> {};
  struct NO : std::bool_constant<false> {};
  /**
   * @brief `nullate::DYNAMIC` defers the determination of nullability to run time rather than
   * compile time. The calling code is responsible for specifying whether or not nulls are
   * present using the constructor parameter at run time.
   */
  struct DYNAMIC {
    DYNAMIC() = delete;
    /**
     * @brief Create a runtime nullate object.
     *
     * @see cudf::column_device_view::optional_begin for example usage
     *
     * @param b True if nulls are expected in the operation in which this
     *          object is applied.
     */
    constexpr explicit DYNAMIC(bool b) noexcept : value{b} {}
    /**
     * @brief Returns true if nulls are expected in the operation in which this object is applied.
     *
     * @return `true` if nulls are expected in the operation in which this object is applied,
     * otherwise false
     */
    constexpr operator bool() const noexcept { return value; }
    bool value;  ///< True if nulls are expected
  };
};

namespace detail {
/**
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 *
 * column_device_view_base and derived classes do not support has_nulls() or
 * null_count().  The primary reason for this is that creation of column_device_views
 * from column_views that have UNKNOWN null counts would require an on-the-spot, and
 * not-obvious computation of null count, which could lead to undesirable performance issues.
 * This information is also generally not needed in device code, and on the host-side
 * is easily accessible from the associated column_view.
 */
class alignas(16) column_device_view_base {
 public:
  column_device_view_base()                               = delete;
  ~column_device_view_base()                              = default;
  column_device_view_base(column_device_view_base const&) = default;  ///< Copy constructor
  column_device_view_base(column_device_view_base&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  column_device_view_base& operator=(column_device_view_base const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  column_device_view_base& operator=(column_device_view_base&&) = default;

  /**
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of
   * a column, and instead, accessing the elements should be done via
   *`data<T>()`.
   *
   * This function will only participate in overload resolution if `is_rep_layout_compatible<T>()`
   * or `std::is_same_v<T,void>` are true.
   *
   * @tparam The type to cast to
   * @return Typed pointer to underlying data
   */
  template <typename T = void,
            CUDF_ENABLE_IF(std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  [[nodiscard]] CUDF_HOST_DEVICE T const* head() const noexcept
  {
    return static_cast<T const*>(_data);
  }

  /**
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * For columns with children, the pointer returned is undefined
   * and should not be used.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data, including the offset
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] CUDF_HOST_DEVICE T const* data() const noexcept
  {
    return head<T>() + _offset;
  }

  /**
   * @brief Returns the number of elements in the column.
   *
   * @return The number of elements in the column
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type size() const noexcept { return _size; }

  /**
   * @brief Returns the element type
   *
   * @return The element type
   */
  [[nodiscard]] CUDF_HOST_DEVICE data_type type() const noexcept { return _type; }

  /**
   * @brief Indicates whether the column can contain null elements, i.e., if it
   *has an allocated bitmask.
   *
   * @note If `null_count() > 0`, this function must always return `true`.
   *
   * @return true The bitmask is allocated
   * @return false The bitmask is not allocated
   */
  [[nodiscard]] CUDF_HOST_DEVICE bool nullable() const noexcept { return nullptr != _null_mask; }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   *
   * @return Raw pointer to the underlying bitmask allocation
   */
  [[nodiscard]] CUDF_HOST_DEVICE bitmask_type const* null_mask() const noexcept
  {
    return _null_mask;
  }

  /**
   * @brief Returns the index of the first element relative to the base memory
   * allocation, i.e., what is returned from `head<T>()`.
   *
   * @return The index of the first element relative to the `head<T>()`
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type offset() const noexcept { return _offset; }

  /**
   * @brief Returns whether the specified element holds a valid value (i.e., not
   * null).
   *
   * Checks first for the existence of the null bitmask. If `nullable() ==
   * false`, this function always returns true.
   *
   * @note If `nullable() == true` can be guaranteed, then it is more performant
   * to use `is_valid_nocheck()`.
   *
   * @param element_index The index of the element to query
   * @return true The element is valid
   * @return false The element is null
   */
  [[nodiscard]] __device__ bool is_valid(size_type element_index) const noexcept
  {
    return not nullable() or is_valid_nocheck(element_index);
  }

  /**
   * @brief Returns whether the specified element holds a valid value (i.e., not
   * null)
   *
   * This function does *not* verify the existence of the bitmask before
   * attempting to read it. Therefore, it is undefined behavior to call this
   * function if `nullable() == false`.
   *
   * @param element_index The index of the element to query
   * @return true The element is valid
   * @return false The element is null
   */
  [[nodiscard]] __device__ bool is_valid_nocheck(size_type element_index) const noexcept
  {
    return bit_is_set(_null_mask, offset() + element_index);
  }

  /**
   * @brief Returns whether the specified element is null.
   *
   * Checks first for the existence of the null bitmask. If `nullable() ==
   * false`, this function always returns false.
   *
   * @note If `nullable() == true` can be guaranteed, then it is more performant
   * to use `is_null_nocheck()`.
   *
   * @param element_index The index of the element to query
   * @return true The element is null
   * @return false The element is valid
   */
  [[nodiscard]] __device__ bool is_null(size_type element_index) const noexcept
  {
    return not is_valid(element_index);
  }

  /**
   * @brief Returns whether the specified element is null
   *
   * This function does *not* verify the existence of the bitmask before
   * attempting to read it. Therefore, it is undefined behavior to call this
   * function if `nullable() == false`.
   *
   * @param element_index The index of the element to query
   * @return true The element is null
   * @return false The element is valid
   */
  [[nodiscard]] __device__ bool is_null_nocheck(size_type element_index) const noexcept
  {
    return not is_valid_nocheck(element_index);
  }

  /**
   * @brief Returns the specified bitmask word from the `null_mask()`.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param word_index The index of the word to get
   * @return bitmask word for the given word_index
   */
  [[nodiscard]] __device__ bitmask_type get_mask_word(size_type word_index) const noexcept
  {
    return null_mask()[word_index];
  }

 protected:
  data_type _type{type_id::EMPTY};   ///< Element type
  cudf::size_type _size{};           ///< Number of elements
  void const* _data{};               ///< Pointer to device memory containing elements
  bitmask_type const* _null_mask{};  ///< Pointer to device memory containing
                                     ///< bitmask representing null elements.
  size_type _offset{};               ///< Index position of the first element.
                                     ///< Enables zero-copy slicing

  /**
   * @brief Constructs a column with the specified type, size, data, nullmask and offset.
   *
   * @param type The type of the column
   * @param size The number of elements in the column
   * @param data Pointer to device memory containing elements
   * @param null_mask Pointer to device memory containing bitmask representing valid elements
   * @param offset Index position of the first element
   */
  CUDF_HOST_DEVICE column_device_view_base(data_type type,
                                           size_type size,
                                           void const* data,
                                           bitmask_type const* null_mask,
                                           size_type offset)
    : _type{type}, _size{size}, _data{data}, _null_mask{null_mask}, _offset{offset}
  {
  }

  template <typename C, typename T, typename = void>
  struct has_element_accessor_impl : std::false_type {};

  template <typename C, typename T>
  struct has_element_accessor_impl<
    C,
    T,
    void_t<decltype(std::declval<C>().template element<T>(std::declval<size_type>()))>>
    : std::true_type {};
};
// @cond
// Forward declaration
template <typename T>
struct value_accessor;
template <typename T, typename Nullate>
struct optional_accessor;
template <typename T, bool has_nulls>
struct pair_accessor;
template <typename T, bool has_nulls>
struct pair_rep_accessor;
template <typename T>
struct mutable_value_accessor;
// @endcond
}  // namespace detail

/**
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 *
 * @ingroup column_classes
 */
class alignas(16) column_device_view : public detail::column_device_view_base {
 public:
  column_device_view()                          = delete;
  ~column_device_view()                         = default;
  column_device_view(column_device_view const&) = default;  ///< Copy constructor
  column_device_view(column_device_view&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  column_device_view& operator=(column_device_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  column_device_view& operator=(column_device_view&&) = default;

  /**
   * @brief Creates an instance of this class using the specified host memory
   * pointer (h_ptr) to store child objects and the device memory pointer
   * (d_ptr) as a base for any child object pointers.
   *
   * @param column Column view from which to create this instance.
   * @param h_ptr Host memory pointer on which to place any child data.
   * @param d_ptr Device memory pointer on which to base any child pointers.
   */
  column_device_view(column_view column, void* h_ptr, void* d_ptr);

  /**
   * @brief Get a new column_device_view which is a slice of this column.
   *
   * Example:
   * @code{.cpp}
   * // column = column_device_view([1, 2, 3, 4, 5, 6, 7])
   * auto c = column.slice(1, 3);
   * // c = column_device_view([2, 3, 4])
   * auto c1 = column.slice(2, 3);
   * // c1 = column_device_view([3, 4, 5])
   * @endcode
   *
   * @param offset The index of the first element in the slice
   * @param size The number of elements in the slice
   * @return A slice of this column
   */
  [[nodiscard]] CUDF_HOST_DEVICE column_device_view slice(size_type offset,
                                                          size_type size) const noexcept
  {
    return column_device_view{this->type(),
                              size,
                              this->head(),
                              this->null_mask(),
                              this->offset() + offset,
                              d_children,
                              this->num_child_columns()};
  }

  /**
   * @brief Returns reference to element at the specified index.
   *
   * If the element at the specified index is NULL, i.e.,
   * `is_null(element_index) == true`, then any attempt to use the result will
   * lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false. Specializations of this function may exist for types `T` where
   *`is_rep_layout_compatible<T>` is false.
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @return reference to the element at the specified index
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    return data<T>()[element_index];
  }

  /**
   * @brief Returns `string_view` to the string element at the specified index.
   *
   * If the element at the specified index is NULL, i.e., `is_null(element_index)
   * == true`, then any attempt to use the result will lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * @param element_index Position of the desired string element
   * @return string_view instance representing this element at this index
   */
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, string_view>)>
  __device__ T element(size_type element_index) const noexcept
  {
    size_type index       = element_index + offset();  // account for this view's _offset
    char const* d_strings = static_cast<char const*>(_data);
    auto const offsets    = d_children[strings_column_view::offsets_column_index];
    auto const itr        = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
    auto const offset     = itr[index];
    return string_view{d_strings + offset, static_cast<cudf::size_type>(itr[index + 1] - offset)};
  }

 private:
  /**
   * @brief Dispatch functor for resolving the index value for a dictionary element.
   *
   * The basic dictionary elements are the indices which can be any index type.
   */
  struct index_element_fn {
    template <typename IndexType,
              CUDF_ENABLE_IF(is_index_type<IndexType>() and std::is_unsigned_v<IndexType>)>
    __device__ size_type operator()(column_device_view const& indices, size_type index)
    {
      return static_cast<size_type>(indices.element<IndexType>(index));
    }

    template <typename IndexType,
              typename... Args,
              CUDF_ENABLE_IF(not(is_index_type<IndexType>() and std::is_unsigned_v<IndexType>))>
    __device__ size_type operator()(Args&&... args)
    {
      CUDF_UNREACHABLE("dictionary indices must be an unsigned integral type");
    }
  };

 public:
  /**
   * @brief Returns `dictionary32` element at the specified index for a
   * dictionary column.
   *
   * `dictionary32` is a strongly typed wrapper around an `int32_t` value that holds the
   * offset into the dictionary keys for the specified element.
   *
   * For example, given a dictionary column `d` with:
   * ```c++
   * keys: {"foo", "bar", "baz"}
   * indices: {2, 0, 2, 1, 0}
   *
   * d.element<dictionary32>(0) == dictionary32{2};
   * d.element<dictionary32>(1) == dictionary32{0};
   * ```
   *
   * If the element at the specified index is NULL, i.e., `is_null(element_index) == true`,
   * then any attempt to use the result will lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * @param element_index Position of the desired element
   * @return dictionary32 instance representing this element at this index
   */
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, dictionary32>)>
  __device__ T element(size_type element_index) const noexcept
  {
    size_type index    = element_index + offset();  // account for this view's _offset
    auto const indices = d_children[0];
    return dictionary32{type_dispatcher(indices.type(), index_element_fn{}, indices, index)};
  }

  /**
   * @brief Returns a `numeric::fixed_point` element at the specified index for a `fixed_point`
   * column.
   *
   * If the element at the specified index is NULL, i.e., `is_null(element_index) == true`,
   * then any attempt to use the result will lead to undefined behavior.
   *
   * @param element_index Position of the desired element
   * @return numeric::fixed_point representing the element at this index
   */
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  __device__ T element(size_type element_index) const noexcept
  {
    using namespace numeric;
    using rep        = typename T::rep;
    auto const scale = scale_type{_type.scale()};
    return T{scaled_integer<rep>{data<rep>()[element_index], scale}};
  }

  /**
   * @brief For a given `T`, indicates if `column_device_view::element<T>()` has a valid overload.
   *
   * @tparam T The element type
   * @return `true` if `column_device_view::element<T>()` has a valid overload, `false` otherwise
   */
  template <typename T>
  static constexpr bool has_element_accessor()
  {
    return has_element_accessor_impl<column_device_view, T>::value;
  }

  /// Counting iterator
  using count_it = thrust::counting_iterator<size_type>;
  /**
   * @brief Iterator for navigating this column
   */
  template <typename T>
  using const_iterator = thrust::transform_iterator<detail::value_accessor<T>, count_it>;

  /**
   * @brief Return an iterator to the first element of the column.
   *
   * This iterator only supports columns where `has_nulls() == false`. Using it
   * with columns where `has_nulls() == true` will result in undefined behavior
   * when accessing null elements.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * For columns with null elements, use `make_null_replacement_iterator`.
   *
   * @tparam T Type of the elements in the column
   * @return An iterator to the first element of the column
   */
  template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_iterator<T> begin() const
  {
    return const_iterator<T>{count_it{0}, detail::value_accessor<T>{*this}};
  }

  /**
   * @brief Returns an iterator to the element following the last element of the column.
   *
   * This iterator only supports columns where `has_nulls() == false`. Using it
   * with columns where `has_nulls() == true` will result in undefined behavior
   * when accessing null elements.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * For columns with null elements, use `make_null_replacement_iterator`.
   *
   * @return An iterator to the element following the last element of the column
   */
  template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_iterator<T> end() const
  {
    return const_iterator<T>{count_it{size()}, detail::value_accessor<T>{*this}};
  }

  /**
   * @brief Optional iterator for navigating this column
   */
  template <typename T, typename Nullate>
  using const_optional_iterator =
    thrust::transform_iterator<detail::optional_accessor<T, Nullate>, count_it>;

  /**
   * @brief Pair iterator for navigating this column
   */
  template <typename T, bool has_nulls>
  using const_pair_iterator =
    thrust::transform_iterator<detail::pair_accessor<T, has_nulls>, count_it>;

  /**
   * @brief Pair rep iterator for navigating this column
   *
   * Each row value is accessed in its representative form.
   */
  template <typename T, bool has_nulls>
  using const_pair_rep_iterator =
    thrust::transform_iterator<detail::pair_rep_accessor<T, has_nulls>, count_it>;

  /**
   * @brief Return an optional iterator to the first element of the column.
   *
   * Dereferencing the returned iterator returns a `cuda::std::optional<T>`.
   *
   * The element of this iterator contextually converts to bool. The conversion returns true
   * if the object contains a value and false if it does not contain a value.
   *
   * Calling this method with `nullate::DYNAMIC` defers the assumption of nullability to
   * runtime with the caller indicating if the column has nulls. The `nullate::DYNAMIC` is
   * useful when an algorithm is going to execute on multiple iterators and all the combinations of
   * iterator types are not required at compile time.
   *
   * @code{.cpp}
   * template<typename T>
   * void some_function(cudf::column_view<T> const& col_view){
   *    auto d_col = cudf::column_device_view::create(col_view);
   *    // Create a `DYNAMIC` optional iterator
   *    auto optional_iterator =
   *       d_col->optional_begin<T>(cudf::nullate::DYNAMIC{col_view.has_nulls()});
   * }
   * @endcode
   *
   * Calling this method with `nullate::YES` means that the column supports nulls and
   * the optional returned might not contain a value.
   *
   * Calling this method with `nullate::NO` means that the column has no null values
   * and the optional returned will always contain a value.
   *
   * @code{.cpp}
   * template<typename T, bool has_nulls>
   * void some_function(cudf::column_view<T> const& col_view){
   *    auto d_col = cudf::column_device_view::create(col_view);
   *    if constexpr(has_nulls) {
   *      auto optional_iterator = d_col->optional_begin<T>(cudf::nullate::YES{});
   *      //use optional_iterator
   *    } else {
   *      auto optional_iterator = d_col->optional_begin<T>(cudf::nullate::NO{});
   *      //use optional_iterator
   *    }
   * }
   * @endcode
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if the column is not nullable and `has_nulls` evaluates to true.
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @tparam T The type of elements in the column
   * @tparam Nullate A cudf::nullate type describing how to check for nulls
   * @param has_nulls  A cudf::nullate type describing how to check for nulls
   * @return An optional iterator to the first element of the column
   */
  template <typename T,
            typename Nullate,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  auto optional_begin(Nullate has_nulls) const
  {
    return const_optional_iterator<T, Nullate>{
      count_it{0}, detail::optional_accessor<T, Nullate>{*this, has_nulls}};
  }

  /**
   * @brief Return a pair iterator to the first element of the column.
   *
   * Dereferencing the returned iterator returns a `thrust::pair<T, bool>`.
   *
   * If an element at position `i` is valid (or `has_nulls == false`), then
   * for `p = *(iter + i)`, `p.first` contains the value of the element at `i`
   * and `p.second == true`.
   *
   * Else, if the element at `i` is null, then the value of `p.first` is
   * undefined and `p.second == false`.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if tparam `has_nulls == true` and
   * `nullable() == false`
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @return A pair iterator to the first element of the column
   */
  template <typename T,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_pair_iterator<T, has_nulls> pair_begin() const
  {
    return const_pair_iterator<T, has_nulls>{count_it{0},
                                             detail::pair_accessor<T, has_nulls>{*this}};
  }

  /**
   * @brief Return a pair iterator to the first element of the column.
   *
   * Dereferencing the returned iterator returns a `thrust::pair<rep_type, bool>`,
   * where `rep_type` is `device_storage_type<T>`, the type used to store
   * the value on the device.
   *
   * If an element at position `i` is valid (or `has_nulls == false`), then
   * for `p = *(iter + i)`, `p.first` contains the value of the element at `i`
   * and `p.second == true`.
   *
   * Else, if the element at `i` is null, then the value of `p.first` is
   * undefined and `p.second == false`.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if tparam `has_nulls == true` and
   * `nullable() == false`
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @return A pair iterator to the first element of the column
   */
  template <typename T,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_pair_rep_iterator<T, has_nulls> pair_rep_begin() const
  {
    return const_pair_rep_iterator<T, has_nulls>{count_it{0},
                                                 detail::pair_rep_accessor<T, has_nulls>{*this}};
  }

  /**
   * @brief Return an optional iterator to the element following the last element of the column.
   *
   * The returned iterator represents a `cuda::std::optional<T>` element.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if the column is not nullable and `has_nulls` is true
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @tparam T The type of elements in the column
   * @tparam Nullate A cudf::nullate type describing how to check for nulls
   * @param has_nulls  A cudf::nullate type describing how to check for nulls
   * @return An optional iterator to the element following the last element of the column
   */
  template <typename T,
            typename Nullate,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  auto optional_end(Nullate has_nulls) const
  {
    return const_optional_iterator<T, Nullate>{
      count_it{size()}, detail::optional_accessor<T, Nullate>{*this, has_nulls}};
  }

  /**
   * @brief Return a pair iterator to the element following the last element of the column.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if tparam `has_nulls == true` and
   * `nullable() == false`
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   * @return A pair iterator to the element following the last element of the column
   */
  template <typename T,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_pair_iterator<T, has_nulls> pair_end() const
  {
    return const_pair_iterator<T, has_nulls>{count_it{size()},
                                             detail::pair_accessor<T, has_nulls>{*this}};
  }

  /**
   * @brief Return a pair iterator to the element following the last element of the column.
   *
   * This function does not participate in overload resolution if
   * `column_device_view::has_element_accessor<T>()` is false.
   *
   * @throws cudf::logic_error if tparam `has_nulls == true` and
   * `nullable() == false`
   * @throws cudf::logic_error if column datatype and Element type mismatch.
   *
   * @return A pair iterator to the element following the last element of the column
   */
  template <typename T,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  [[nodiscard]] const_pair_rep_iterator<T, has_nulls> pair_rep_end() const
  {
    return const_pair_rep_iterator<T, has_nulls>{count_it{size()},
                                                 detail::pair_rep_accessor<T, has_nulls>{*this}};
  }

  /**
   * @brief Factory to construct a column view that is usable in device memory.
   *
   * Allocates and copies views of `source_view`'s children to device memory to
   * make them accessible in device code.
   *
   * If `source_view.num_children() == 0`, then no device memory is allocated.
   *
   * Returns a `std::unique_ptr<column_device_view>` with a custom deleter to
   * free the device memory allocated for the children.
   *
   * A `column_device_view` should be passed by value into GPU kernels.
   *
   * @param source_view The `column_view` to make usable in device code
   * @param stream CUDA stream used for device memory operations for children columns.
   * @return A `unique_ptr` to a `column_device_view` that makes the data from
   *`source_view` available in device memory.
   */
  static std::unique_ptr<column_device_view, std::function<void(column_device_view*)>> create(
    column_view source_view, rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Destroy the `column_device_view` object.
   *
   * @note Does not free the column data, simply frees the device memory
   * allocated to hold the child views.
   */
  void destroy();

  /**
   * @brief Return the size in bytes of the amount of memory needed to hold a
   * device view of the specified column and it's children.
   *
   * @param source_view The `column_view` to use for this calculation.
   * @return number of bytes to store device view in GPU memory
   */
  static std::size_t extent(column_view const& source_view);

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return column_view The requested child `column_view`
   */
  [[nodiscard]] __device__ column_device_view child(size_type child_index) const noexcept
  {
    return d_children[child_index];
  }

  /**
   * @brief Returns a span containing the children of this column
   *
   * @return A span containing the children of this column
   */
  [[nodiscard]] __device__ device_span<column_device_view const> children() const noexcept
  {
    return device_span<column_device_view const>(d_children, _num_children);
  }

  /**
   * @brief Returns the number of child columns
   *
   * @return The number of child columns
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type num_child_columns() const noexcept
  {
    return _num_children;
  }

 private:
  /**
   * @brief Creates an instance of this class using pre-existing device memory pointers to data,
   * nullmask, and offset.
   *
   * @param type The type of the column
   * @param size The number of elements in the column
   * @param data Pointer to the device memory containing the data
   * @param null_mask Pointer to the device memory containing the null bitmask
   * @param offset The index of the first element in the column
   * @param children Pointer to the device memory containing child data
   * @param num_children The number of child columns
   */
  CUDF_HOST_DEVICE column_device_view(data_type type,
                                      size_type size,
                                      void const* data,
                                      bitmask_type const* null_mask,
                                      size_type offset,
                                      column_device_view* children,
                                      size_type num_children)
    : column_device_view_base(type, size, data, null_mask, offset),
      d_children(children),
      _num_children(num_children)
  {
  }

 protected:
  column_device_view* d_children{};  ///< Array of `column_device_view`
                                     ///< objects in device memory.
                                     ///< Based on element type, children
                                     ///< may contain additional data
  size_type _num_children{};         ///< The number of child columns

  /**
   * @brief Construct's a `column_device_view` from a `column_view` populating
   * all but the children.
   *
   * @note This constructor is for internal use only. To create a
   *`column_device_view` from a `column_view`, the
   *`column_device_view::create()` function should be used.
   *
   * @param source The `column_view` to use for this construction
   */
  column_device_view(column_view source);
};

/**
 * @brief A mutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 *
 * @ingroup column_classes
 */
class alignas(16) mutable_column_device_view : public detail::column_device_view_base {
 public:
  mutable_column_device_view()                                  = delete;
  ~mutable_column_device_view()                                 = default;
  mutable_column_device_view(mutable_column_device_view const&) = default;  ///< Copy constructor
  mutable_column_device_view(mutable_column_device_view&&)      = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  mutable_column_device_view& operator=(mutable_column_device_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  mutable_column_device_view& operator=(mutable_column_device_view&&) = default;

  /**
   * @brief Creates an instance of this class using the specified host memory
   * pointer (h_ptr) to store child objects and the device memory pointer
   * (d_ptr) as a base for any child object pointers.
   *
   * @param column Column view from which to create this instance.
   * @param h_ptr Host memory pointer on which to place any child data.
   * @param d_ptr Device memory pointer on which to base any child pointers.
   */
  mutable_column_device_view(mutable_column_view column, void* h_ptr, void* d_ptr);

  /**
   * @brief Factory to construct a column view that is usable in device memory.
   *
   * Allocates and copies views of `source_view`'s children to device memory to
   * make them accessible in device code.
   *
   * If `source_view.num_children() == 0`, then no device memory is allocated.
   *
   * Returns a `std::unique_ptr<mutable_column_device_view>` with a custom
   * deleter to free the device memory allocated for the children.
   *
   * A `mutable_column_device_view` should be passed by value into GPU kernels.
   *
   * @param source_view The `column_view` to make usable in device code
   * @param stream CUDA stream used for device memory operations for children columns.
   * @return A `unique_ptr` to a `mutable_column_device_view` that makes the
   * data from `source_view` available in device memory.
   */
  static std::unique_ptr<mutable_column_device_view,
                         std::function<void(mutable_column_device_view*)>>
  create(mutable_column_view source_view,
         rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * This function will only participate in overload resolution if `is_rep_layout_compatible<T>()`
   * or `std::is_same_v<T,void>` are true.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of
   * a column, and instead, accessing the elements should be done via
   * `data<T>()`.
   *
   * @tparam The type to cast to
   * @return Typed pointer to underlying data
   */
  template <typename T = void,
            CUDF_ENABLE_IF(std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  CUDF_HOST_DEVICE T* head() const noexcept
  {
    return const_cast<T*>(detail::column_device_view_base::head<T>());
  }

  /**
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @tparam T The type to cast to
   * @return Typed pointer to underlying data, including the offset
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  CUDF_HOST_DEVICE T* data() const noexcept
  {
    return const_cast<T*>(detail::column_device_view_base::data<T>());
  }

  /**
   * @brief Returns reference to element at the specified index.
   *
   * This function accounts for the offset.
   *
   * This function does not participate in overload resolution if `is_rep_layout_compatible<T>` is
   * false. Specializations of this function may exist for types `T` where
   *`is_rep_layout_compatible<T>` is false.
   *
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @return Reference to the element at the specified index
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  __device__ T& element(size_type element_index) const noexcept
  {
    return data<T>()[element_index];
  }

  /**
   * @brief For a given `T`, indicates if `mutable_column_device_view::element<T>()` has a valid
   * overload.
   *
   * @return `true` if `mutable_column_device_view::element<T>()` has a valid overload, `false`
   */
  template <typename T>
  static constexpr bool has_element_accessor()
  {
    return has_element_accessor_impl<mutable_column_device_view, T>::value;
  }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   * @return Raw pointer to the underlying bitmask allocation
   */
  [[nodiscard]] CUDF_HOST_DEVICE bitmask_type* null_mask() const noexcept
  {
    return const_cast<bitmask_type*>(detail::column_device_view_base::null_mask());
  }

  /// Counting iterator
  using count_it = thrust::counting_iterator<size_type>;
  /**
   * @brief Iterator for navigating this column
   */
  template <typename T>
  using iterator = thrust::transform_iterator<detail::mutable_value_accessor<T>, count_it>;

  /**
   * @brief Return first element (accounting for offset) after underlying data
   * is casted to the specified type.
   *
   * This function does not participate in overload resolution if
   * `mutable_column_device_view::has_element_accessor<T>()` is false.
   *
   * @tparam T The desired type
   * @return Pointer to the first element after casting
   */
  template <typename T, CUDF_ENABLE_IF(mutable_column_device_view::has_element_accessor<T>())>
  iterator<T> begin()
  {
    return iterator<T>{count_it{0}, detail::mutable_value_accessor<T>{*this}};
  }

  /**
   * @brief Return one past the last element after underlying data is casted to
   * the specified type.
   *
   * This function does not participate in overload resolution if
   * `mutable_column_device_view::has_element_accessor<T>()` is false.
   *
   * @tparam T The desired type
   * @return Pointer to one past the last element after casting
   */
  template <typename T, CUDF_ENABLE_IF(mutable_column_device_view::has_element_accessor<T>())>
  iterator<T> end()
  {
    return iterator<T>{count_it{size()}, detail::mutable_value_accessor<T>{*this}};
  }

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return The requested child `column_view`
   */
  [[nodiscard]] __device__ mutable_column_device_view child(size_type child_index) const noexcept
  {
    return d_children[child_index];
  }

#ifdef __CUDACC__  // because set_bit in bit.hpp is wrapped with __CUDACC__
  /**
   * @brief Updates the null mask to indicate that the specified element is
   * valid
   *
   * @note This operation requires a global atomic operation. Therefore, it is
   * not recommended to use this function in performance critical regions. When
   * possible, it is more efficient to compute and update an entire word at
   * once using `set_mask_word`.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   */
  __device__ void set_valid(size_type element_index) const noexcept
  {
    return set_bit(null_mask(), element_index);
  }

  /**
   * @brief Updates the null mask to indicate that the specified element is null
   *
   * @note This operation requires a global atomic operation. Therefore, it is
   * not recommended to use this function in performance critical regions. When
   * possible, it is more efficient to compute and update an entire word at
   * once using `set_mask_word`.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   */
  __device__ void set_null(size_type element_index) const noexcept
  {
    return clear_bit(null_mask(), element_index);
  }

#endif

  /**
   * @brief Updates the specified bitmask word in the `null_mask()` with a
   * new word.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param word_index The index of the word to update
   * @param new_word The new bitmask word
   */
  __device__ void set_mask_word(size_type word_index, bitmask_type new_word) const noexcept
  {
    null_mask()[word_index] = new_word;
  }

  /**
   * @brief Return the size in bytes of the amount of memory needed to hold a
   * device view of the specified column and it's children.
   *
   * @param source_view The `column_view` to use for this calculation.
   * @return The size in bytes of the amount of memory needed to hold a
   * device view of the specified column and it's children
   */
  static std::size_t extent(mutable_column_view source_view);

  /**
   * @brief Destroy the `mutable_column_device_view` object.
   *
   * @note Does not free the column data, simply frees the device memory
   * allocated to hold the child views.
   */
  void destroy();

 private:
  mutable_column_device_view* d_children{};  ///< Array of `mutable_column_device_view`
                                             ///< objects in device memory.
                                             ///< Based on element type, children
                                             ///< may contain additional data
  size_type _num_children{};                 ///< The number of child columns

  /**
   * @brief Construct's a `mutable_column_device_view` from a
   *`mutable_column_view` populating all but the children.
   *
   * @note This constructor is for internal use only. To create a
   *`mutable_column_device_view` from a `column_view`, the
   *`mutable_column_device_view::create()` function should be used.
   */
  mutable_column_device_view(mutable_column_view source);
};

namespace detail {

#ifdef __CUDACC__  // because set_bit in bit.hpp is wrapped with __CUDACC__

/**
 * @brief Convenience function to get offset word from a bitmask
 *
 * @see copy_offset_bitmask
 * @see offset_bitmask_binop
 */
__device__ inline bitmask_type get_mask_offset_word(bitmask_type const* __restrict__ source,
                                                    size_type destination_word_index,
                                                    size_type source_begin_bit,
                                                    size_type source_end_bit)
{
  size_type source_word_index = destination_word_index + word_index(source_begin_bit);
  bitmask_type curr_word      = source[source_word_index];
  bitmask_type next_word      = 0;
  if (word_index(source_end_bit - 1) >
      word_index(source_begin_bit +
                 destination_word_index * detail::size_in_bits<bitmask_type>())) {
    next_word = source[source_word_index + 1];
  }
  return __funnelshift_r(curr_word, next_word, source_begin_bit);
}

#endif

/**
 * @brief value accessor of column without null bitmask
 *
 * A unary functor returns scalar value at `id`.
 * `operator() (cudf::size_type id)` computes `element`
 * This functor is only allowed for non-nullable columns.
 *
 * the return value for element `i` will return `column[i]`
 *
 * @throws cudf::logic_error if the column is nullable.
 * @throws cudf::logic_error if column datatype and template T type mismatch.
 *
 * @tparam T The type of elements in the column
 */
template <typename T>
struct value_accessor {
  column_device_view const col;  ///< column view of column in device

  /**
   * @brief constructor
   *
   * @param[in] _col column device view of cudf column
   */
  value_accessor(column_device_view const& _col) : col{_col}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
  }

  /**
   * @brief Returns the value of element at index `i`
   * @param[in] i index of element
   * @return value of element at index `i`
   */
  __device__ T operator()(cudf::size_type i) const { return col.element<T>(i); }
};

/**
 * @brief optional accessor of a column
 *
 *
 * The optional_accessor always returns a `thrust::optional` of `column[i]`. The validity
 * of the optional is determined by the `Nullate` parameter which may be one of the following:
 *
 * - `nullate::YES` means that the column supports nulls and the optional returned
 *    might be valid or invalid.
 *
 * - `nullate::NO` means the caller attests that the column has no null values,
 *    no checks will occur and `thrust::optional{column[i]}` will be
 *    return for each `i`.
 *
 * - `nullate::DYNAMIC` defers the assumption of nullability to runtime and the caller
 *    specifies if the column has nulls at runtime.
 *    For `DYNAMIC{true}` the return value will be `thrust::optional{column[i]}` if
 *      element `i` is not null and `thrust::optional{}` if element `i` is null.
 *    For `DYNAMIC{false}` the return value will always be `thrust::optional{column[i]}`.
 *
 * @throws cudf::logic_error if column datatype and template T type mismatch.
 * @throws cudf::logic_error if the column is not nullable and `with_nulls` evaluates to true
 *
 * @tparam T The type of elements in the column
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename T, typename Nullate>
struct optional_accessor {
  column_device_view const col;  ///< column view of column in device

  /**
   * @brief Constructor
   *
   * @param _col Column on which to iterator over its elements.
   * @param with_nulls Indicates if the `col` should be checked for nulls.
   */
  optional_accessor(column_device_view const& _col, Nullate with_nulls)
    : col{_col}, has_nulls{with_nulls}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
    if (with_nulls) { CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column."); }
  }

  /**
   * @brief Returns a `thrust::optional` of `column[i]`.
   *
   * @param i The index of the element to return
   * @return A `thrust::optional` that contains the value of `column[i]` is not null. If that
   * element is null, the resulting optional will not contain a value.
   */
  __device__ inline cuda::std::optional<T> operator()(cudf::size_type i) const
  {
    if (has_nulls) {
      return (col.is_valid_nocheck(i)) ? cuda::std::optional<T>{col.element<T>(i)}
                                       : cuda::std::optional<T>{cuda::std::nullopt};
    }
    return cuda::std::optional<T>{col.element<T>(i)};
  }

  Nullate has_nulls{};  ///< Indicates if the `col` should be checked for nulls.
};

/**
 * @brief pair accessor of column with/without null bitmask
 *
 * A unary functor returns pair with scalar value at `id` and boolean validity
 * `operator() (cudf::size_type id)` computes `element`  and
 * returns a `pair(element, validity)`
 *
 * the return value for element `i` will return `pair(column[i], validity)`
 * `validity` is `true` if `has_nulls=false`.
 * `validity` is validity of the element at `i` if `has_nulls=true` and the
 * column is nullable.
 *
 * @throws cudf::logic_error if `has_nulls==true` and the column is not
 * nullable.
 * @throws cudf::logic_error if column datatype and template T type mismatch.
 *
 * @tparam T The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 */
template <typename T, bool has_nulls = false>
struct pair_accessor {
  column_device_view const col;  ///< column view of column in device

  /**
   * @brief constructor
   *
   * @param[in] _col column device view of cudf column
   */
  pair_accessor(column_device_view const& _col) : col{_col}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
    if (has_nulls) { CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column."); }
  }

  /**
   * @brief Pair accessor
   *
   * @param[in] i index of the element
   * @return pair(element, validity)
   */
  __device__ inline thrust::pair<T, bool> operator()(cudf::size_type i) const
  {
    return {col.element<T>(i), (has_nulls ? col.is_valid_nocheck(i) : true)};
  }
};

/**
 * @brief pair accessor of column with/without null bitmask
 *
 * A unary functor returns pair with representative scalar value at `id` and boolean validity
 * `operator() (cudf::size_type id)` computes `element`  and
 * returns a `pair(element, validity)`
 *
 * the return value for element `i` will return `pair(column[i], validity)`
 * `validity` is `true` if `has_nulls=false`.
 * `validity` is validity of the element at `i` if `has_nulls=true` and the
 * column is nullable.
 *
 * @throws cudf::logic_error if `has_nulls==true` and the column is not
 * nullable.
 * @throws cudf::logic_error if column datatype and template T type mismatch.
 *
 * @tparam T The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 */
template <typename T, bool has_nulls = false>
struct pair_rep_accessor {
  column_device_view const col;  ///< column view of column in device

  using rep_type = device_storage_type_t<T>;  ///< representation type

  /**
   * @brief constructor
   *
   * @param[in] _col column device view of cudf column
   */
  pair_rep_accessor(column_device_view const& _col) : col{_col}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
    if (has_nulls) { CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column."); }
  }

  /**
   * @brief Pair accessor
   *
   * @param[in] i index of element to access
   * @return pair of element and validity
   */
  __device__ inline thrust::pair<rep_type, bool> operator()(cudf::size_type i) const
  {
    return {get_rep<T>(i), (has_nulls ? col.is_valid_nocheck(i) : true)};
  }

 private:
  template <typename R, std::enable_if_t<std::is_same_v<R, rep_type>, void>* = nullptr>
  __device__ inline auto get_rep(cudf::size_type i) const
  {
    return col.element<R>(i);
  }

  template <typename R, std::enable_if_t<not std::is_same_v<R, rep_type>, void>* = nullptr>
  __device__ inline auto get_rep(cudf::size_type i) const
  {
    return col.element<R>(i).value();
  }
};

/**
 * @brief Mutable value accessor of column without null bitmask
 *
 * A unary functor that accepts an index and returns a reference to the element at that index in the
 * column.
 *
 * @throws cudf::logic_error if the column is nullable
 * @throws cudf::logic_error if column datatype and template T type mismatch
 *
 * @tparam T The type of elements in the column
 */
template <typename T>
struct mutable_value_accessor {
  mutable_column_device_view col;  ///< mutable column view of column in device

  /**
   * @brief Constructor
   *
   * @param[in] _col mutable column device view of cudf column
   */
  mutable_value_accessor(mutable_column_device_view& _col) : col{_col}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<T>(col.type().id()), "the data type mismatch");
  }

  /**
   * @brief Accessor
   *
   * @param i index of element to access
   * @return reference to element at `i`
   */
  __device__ T& operator()(cudf::size_type i) { return col.element<T>(i); }
};

/**
 * @brief Helper function for use by column_device_view and mutable_column_device_view
 * constructors to build device_views from views.
 *
 * It is used to build the array of child columns in device memory. Since child columns can
 * also have child columns, this uses recursion to build up the flat device buffer to contain
 * all the children and set the member pointers appropriately.
 *
 * This is accomplished by laying out all the children and grand-children into a flat host
 * buffer first but also keep a running device pointer to use when setting the
 * d_children array result.
 *
 * This function is provided both the host pointer in which to insert its children (and
 * by recursion its grand-children) and the device pointer to be used when calculating
 * ultimate device pointer for the d_children member.
 *
 * @tparam ColumnView is either column_view or mutable_column_view
 * @tparam ColumnDeviceView is either column_device_view or mutable_column_device_view
 *
 * @param child_begin Iterator pointing to begin of child columns to make into a device view
 * @param child_end   Iterator pointing to end   of child columns to make into a device view
 * @param h_ptr The host memory where to place any child data
 * @param d_ptr The device pointer for calculating the d_children member of any child data
 * @return The device pointer to be used for the d_children member of the given column
 */
template <typename ColumnDeviceView, typename ColumnViewIterator>
ColumnDeviceView* child_columns_to_device_array(ColumnViewIterator child_begin,
                                                ColumnViewIterator child_end,
                                                void* h_ptr,
                                                void* d_ptr)
{
  ColumnDeviceView* d_children = detail::align_ptr_for_type<ColumnDeviceView>(d_ptr);
  auto num_children            = std::distance(child_begin, child_end);
  if (num_children > 0) {
    // The beginning of the memory must be the fixed-sized ColumnDeviceView
    // struct objects in order for d_children to be used as an array.
    auto h_column = detail::align_ptr_for_type<ColumnDeviceView>(h_ptr);
    auto d_column = d_children;

    // Any child data is assigned past the end of this array: h_end and d_end.
    auto h_end = reinterpret_cast<int8_t*>(h_column + num_children);
    auto d_end = reinterpret_cast<int8_t*>(d_column + num_children);
    std::for_each(child_begin, child_end, [&](auto const& col) {
      // inplace-new each child into host memory
      new (h_column) ColumnDeviceView(col, h_end, d_end);
      h_column++;  // advance to next child
      // update the pointers for holding this child column's child data
      auto col_child_data_size = ColumnDeviceView::extent(col) - sizeof(ColumnDeviceView);
      h_end += col_child_data_size;
      d_end += col_child_data_size;
    });
  }
  return d_children;
}

}  // namespace detail
}  // namespace cudf
