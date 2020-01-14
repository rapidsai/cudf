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

#include <cudf/column/column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {

namespace detail {

/**---------------------------------------------------------------------------*
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 *---------------------------------------------------------------------------**/
class alignas(16) column_device_view_base {
 public:
  column_device_view_base() = delete;
  ~column_device_view_base() = default;
  column_device_view_base(column_device_view_base const&) = default;
  column_device_view_base(column_device_view_base&&) = default;
  column_device_view_base& operator=(column_device_view_base const&) = default;
  column_device_view_base& operator=(column_device_view_base&&) = default;

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of
   * a column, and instead, accessing the elements should be done via
   *`data<T>()`.
   *
   * @tparam The type to cast to
   * @return T const* Typed pointer to underlying data
   *---------------------------------------------------------------------------**/
  template <typename T = void>
  __host__ __device__ T const* head() const noexcept {
    return static_cast<T const*>(_data);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @TODO Clarify behavior for variable-width types.
   *
   * @tparam T The type to cast to
   * @return T const* Typed pointer to underlying data, including the offset
   *---------------------------------------------------------------------------**/
  template <typename T>
  __host__ __device__ T const* data() const noexcept {
    return head<T>() + _offset;
  }
 
  /**---------------------------------------------------------------------------*
   * @brief Returns the number of elements in the column
   *---------------------------------------------------------------------------**/
  __host__ __device__ size_type size() const noexcept { return _size; }

  /**---------------------------------------------------------------------------*
   * @brief Returns the element type
   *---------------------------------------------------------------------------**/
  __host__ __device__ data_type type() const noexcept { return _type; }

  /**---------------------------------------------------------------------------*
   * @brief Indicates whether the column can contain null elements, i.e., if it
   *has an allocated bitmask.
   *
   * @note If `null_count() > 0`, this function must always return `true`.
   *
   * @return true The bitmask is allocated
   * @return false The bitmask is not allocated
   *---------------------------------------------------------------------------**/
  __host__ __device__ bool nullable() const noexcept {
    return nullptr != _null_mask;
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the count of null elements
   *---------------------------------------------------------------------------**/
  __host__ __device__ size_type null_count() const noexcept {
    return _null_count;
  }

  /**---------------------------------------------------------------------------*
   * @brief Indicates whether the column contains null elements,
   * i.e., `null_count() > 0`
   *
   * @return true The column contains null elements
   * @return false All elements are valid
   *---------------------------------------------------------------------------**/
  __host__ __device__ bool has_nulls() const noexcept {
    return null_count() > 0;
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   *---------------------------------------------------------------------------**/
  __host__ __device__ bitmask_type const* null_mask() const noexcept {
    return _null_mask;
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the index of the first element relative to the base memory
   * allocation, i.e., what is returned from `head<T>()`.
   *---------------------------------------------------------------------------**/
  __host__ __device__ size_type offset() const noexcept { return _offset; }

  /**---------------------------------------------------------------------------*
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
   *---------------------------------------------------------------------------**/
  __device__ bool is_valid(size_type element_index) const noexcept {
    return not nullable() or is_valid_nocheck(element_index);
  }

  /**---------------------------------------------------------------------------*
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
   *---------------------------------------------------------------------------**/
  __device__ bool is_valid_nocheck(size_type element_index) const noexcept {
    return bit_is_set(_null_mask, offset()+element_index);
  }

  /**---------------------------------------------------------------------------*
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
   *---------------------------------------------------------------------------**/
  __device__ bool is_null(size_type element_index) const noexcept {
    return not is_valid(element_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns whether the specified element is null
   *
   * This function does *not* verify the existence of the bitmask before
   * attempting to read it. Therefore, it is undefined behavior to call this
   * function if `nullable() == false`.
   *
   * @param element_index The index of the element to query
   * @return true The element is null
   * @return false The element is valid
   *---------------------------------------------------------------------------**/
  __device__ bool is_null_nocheck(size_type element_index) const noexcept {
    return not is_valid_nocheck(element_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the the specified bitmask word from the `null_mask()`.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index
   * @return __device__ get_mask_word
   *---------------------------------------------------------------------------**/
  __device__ bitmask_type get_mask_word(size_type word_index) const noexcept {
    return null_mask()[word_index];
  }

 protected:
  data_type _type{EMPTY};   ///< Element type
  cudf::size_type _size{};  ///< Number of elements
  void const* _data{};      ///< Pointer to device memory containing elements
  bitmask_type const* _null_mask{};  ///< Pointer to device memory containing
                                     ///< bitmask representing null elements.
                                     ///< Optional if `null_count() == 0`
  size_type _null_count{};           ///< The number of null elements
  size_type _offset{};               ///< Index position of the first element.
                                     ///< Enables zero-copy slicing

  column_device_view_base(data_type type, size_type size, void const* data,
                          bitmask_type const* null_mask, size_type null_count,
                          size_type offset)
      : _type{type},
        _size{size},
        _data{data},
        _null_mask{null_mask},
        _null_count{null_count},
        _offset{offset} {}
};

//Forward declaration
template <typename T>
struct value_accessor; 
template <typename T, bool has_nulls>
struct pair_accessor; 
template <typename T>
struct mutable_value_accessor;
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 *---------------------------------------------------------------------------**/
class alignas(16) column_device_view : public detail::column_device_view_base {
 public:
  column_device_view() = delete;
  ~column_device_view() = default;
  column_device_view(column_device_view const&) = default;
  column_device_view(column_device_view&&) = default;
  column_device_view& operator=(column_device_view const&) = default;
  column_device_view& operator=(column_device_view&&) = default;

  /**---------------------------------------------------------------------------*
   * @brief Creates an instance of this class using the specified host memory
   * pointer (h_ptr) to store child objects and the device memory pointer
   *(d_ptr) as a base for any child object pointers.
   *
   * @param column Column view from which to create this instance.
   * @param h_ptr Host memory pointer on which to place any child data.
   * @param d_ptr Device memory pointer on which to base any child pointers.
   *---------------------------------------------------------------------------**/
  column_device_view(column_view column, ptrdiff_t h_ptr, ptrdiff_t d_ptr);

  /**---------------------------------------------------------------------------*
   * @brief Returns reference to element at the specified index.
   *
   * If the element at the specified index is NULL, i.e.,
   *`is_null(element_index) == true`, then any attempt to use the result will
   *lead to undefined behavior.
   *
   * This function accounts for the offset.
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   *---------------------------------------------------------------------------**/
  template <typename T>
  __device__ T const element(size_type element_index) const noexcept {
    return data<T>()[element_index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Iterator for navigating this column
   *---------------------------------------------------------------------------**/
  using count_it = thrust::counting_iterator<size_type>;
  template <typename T>
  using const_iterator =
      thrust::transform_iterator<detail::value_accessor<T>, count_it>;

  /**---------------------------------------------------------------------------*
   * @brief Return an iterator to the first element of the column.
   * 
   * This iterator only supports columns where `has_nulls() == false`. 
   * For columns with null elements, use `make_null_replacement_iterator`.
   * @throws `cudf::logic_error` if `has_nulls() == true`
   *---------------------------------------------------------------------------**/
  template <typename T>
  const_iterator<T> begin() const {
    return const_iterator<T>{count_it{0}, detail::value_accessor<T>{*this}};
  }

  /**---------------------------------------------------------------------------*
   * @brief Return an iterator to the element following the last element of the column. 
   *
   * This iterator only supports columns where `has_nulls() == false`.
   * For columns with null elements, use `make_null_replacement_iterator`.
   * @throws `cudf::logic_error` if `has_nulls() == true`
   *---------------------------------------------------------------------------**/
  template<typename T>
  const_iterator<T> end() const {
    return const_iterator<T>{count_it{size()}, detail::value_accessor<T>{*this}};
  }

  /**---------------------------------------------------------------------------*
   * @brief Pair iterator for navigating this column
   *---------------------------------------------------------------------------**/
  template <typename T, bool has_nulls>
  using const_pair_iterator =
      thrust::transform_iterator<detail::pair_accessor<T, has_nulls>, count_it>;

  /**---------------------------------------------------------------------------*
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
   * @throws `cudf::logic_error` if tparam `has_nulls == true` and 
   * `has_nulls() == false`
   * @throws `cudf::logic_error` if column datatype and Element type mismatch.
   *---------------------------------------------------------------------------**/
  template <typename T, bool has_nulls>
  const_pair_iterator<T,has_nulls> pair_begin() const {
    return const_pair_iterator<T, has_nulls>{
        count_it{0}, detail::pair_accessor<T, has_nulls>{*this}};
  }

  /**---------------------------------------------------------------------------*
   * @brief Return a pair iterator to the element following the last element of
   * the column.
   *
   * @throws `cudf::logic_error` if tparam `has_nulls == true` and 
   * `has_nulls() == false`
   * @throws `cudf::logic_error` if column datatype and Element type mismatch.
   *---------------------------------------------------------------------------**/
  template<typename T, bool has_nulls>
  const_pair_iterator<T, has_nulls> pair_end() const {
    return const_pair_iterator<T, has_nulls>{
        count_it{size()}, detail::pair_accessor<T, has_nulls>{*this}};
  }

  /**---------------------------------------------------------------------------*
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
   * @param stream optional, stream on which the memory for children will be
   * allocated
   * @return A `unique_ptr` to a `column_device_view` that makes the data from
   *`source_view` available in device memory.
   *---------------------------------------------------------------------------**/
  static std::unique_ptr<column_device_view,
                         std::function<void(column_device_view*)>>
  create(column_view source_view, cudaStream_t stream = 0);

  /**---------------------------------------------------------------------------*
   * @brief Destroy the `device_column_view` object.
   *
   * @note Does not free the column data, simply free's the device memory
   * allocated to hold the child views.
   *---------------------------------------------------------------------------**/
  void destroy();

  /**---------------------------------------------------------------------------*
   * @brief Return the size in bytes of the amount of memory needed to hold a
   * device view of the specified column and it's children.
   *
   * @param source_view The `column_view` to use for this calculation.
   *---------------------------------------------------------------------------**/
  static size_type extent(column_view source_view);

  /**---------------------------------------------------------------------------*
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return column_view The requested child `column_view`
   *---------------------------------------------------------------------------**/
  __device__ column_device_view child(size_type child_index) const noexcept {
    return d_children[child_index];
  }

 protected:
  column_device_view* d_children{};  ///< Array of `column_device_view`
                                     ///< objects in device memory.
                                     ///< Based on element type, children
                                     ///< may contain additional data
  size_type _num_children{};         ///< The number of child columns

  /**---------------------------------------------------------------------------*
   * @brief Construct's a `column_device_view` from a `column_view` populating
   * all but the children.
   *
   * @note This constructor is for internal use only. To create a
   *`column_device_view` from a `column_view`, the
   *`column_device_view::create()` function should be used.
   *---------------------------------------------------------------------------**/
  column_device_view(column_view source);
};

/**---------------------------------------------------------------------------*
 * @brief A mutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable in CUDA device code.
 *---------------------------------------------------------------------------**/
class alignas(16) mutable_column_device_view
    : public detail::column_device_view_base {
 public:
  mutable_column_device_view() = delete;
  ~mutable_column_device_view() = default;
  mutable_column_device_view(mutable_column_device_view const&) = default;
  mutable_column_device_view(mutable_column_device_view&&) = default;
  mutable_column_device_view& operator=(mutable_column_device_view const&) =
      default;
  mutable_column_device_view& operator=(mutable_column_device_view&&) = default;

  /**---------------------------------------------------------------------------*
   * @brief Creates an instance of this class using the specified host memory
   * pointer (h_ptr) to store child objects and the device memory pointer
   *(d_ptr) as a base for any child object pointers.
   *
   * @param column Column view from which to create this instance.
   * @param h_ptr Host memory pointer on which to place any child data.
   * @param d_ptr Device memory pointer on which to base any child pointers.
   *---------------------------------------------------------------------------**/
  mutable_column_device_view(mutable_column_view column, ptrdiff_t h_ptr,
                             ptrdiff_t d_ptr);

  /**---------------------------------------------------------------------------*
   * @brief Factory to construct a column view that is usable in device memory.
   *
   * Allocates and copies views of `soure_view`'s children to device memory to
   * make them accessible in device code.
   *
   * If `source_view.num_children() == 0`, then no device memory is allocated.
   *
   * Returns a `std::unique_ptr<mutable_column_device_view>` with a custom
   *deleter to free the device memory allocated for the children.
   *
   * A `mutable_column_device_view` should be passed by value into GPU kernels.
   *
   * @param source_view The `column_view` to make usable in device code
   * @param stream optional, stream on which the memory for children will be
   * allocated
   * @return A `unique_ptr` to a `mutable_column_device_view` that makes the
   *data from `source_view` available in device memory.
   *---------------------------------------------------------------------------**/
  static std::unique_ptr<mutable_column_device_view,
                         std::function<void(mutable_column_device_view*)>>
  create(mutable_column_view source_view, cudaStream_t stream = 0);

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to the base device memory allocation casted to
   * the specified type.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @note It should be rare to need to access the `head<T>()` allocation of
   *a column, and instead, accessing the elements should be done via
   *`data<T>()`.
   *
   * @tparam The type to cast to
   * @return T* Typed pointer to underlying data
   *---------------------------------------------------------------------------**/
  template <typename T = void>
  __host__ __device__ T* head() const noexcept {
    return const_cast<T*>(detail::column_device_view_base::head<T>());
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the underlying data casted to the specified type, plus the
   * offset.
   *
   * @note If `offset() == 0`, then `head<T>() == data<T>()`
   *
   * @TODO Clarify behavior for variable-width types.
   *
   * @tparam T The type to cast to
   * @return T* Typed pointer to underlying data, including the offset
   *---------------------------------------------------------------------------**/
  template <typename T>
  __host__ __device__ T* data() const noexcept {
    return const_cast<T*>(detail::column_device_view_base::data<T>());
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns reference to element at the specified index.
   *
   * This function accounts for the offset.
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   *---------------------------------------------------------------------------**/
  template <typename T>
  __device__ T& element(size_type element_index) noexcept {
    return data<T>()[element_index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note This function does *not* account for the `offset()`.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   *---------------------------------------------------------------------------**/
  __host__ __device__ bitmask_type* null_mask() const noexcept {
    return const_cast<bitmask_type*>(
        detail::column_device_view_base::null_mask());
  }

  /**---------------------------------------------------------------------------*
   * @brief Iterator for navigating this column
   *---------------------------------------------------------------------------**/
  using count_it = thrust::counting_iterator<size_type>;
  template <typename T>
  using iterator =
      thrust::transform_iterator<detail::mutable_value_accessor<T>, count_it>;

  /**---------------------------------------------------------------------------*
   * @brief Return first element (accounting for offset) after underlying data
   * is casted to the specified type.
   *
   * @tparam T The desired type
   * @return T* Pointer to the first element after casting
   *---------------------------------------------------------------------------**/
  template <typename T>
  std::enable_if_t<is_fixed_width<T>(), iterator<T>>
  begin() {
    return iterator<T>{count_it{0}, detail::mutable_value_accessor<T>{*this}};
  }

  /**---------------------------------------------------------------------------*
   * @brief Return one past the last element after underlying data is casted to
   * the specified type.
   *
   * @tparam T The desired type
   * @return T const* Pointer to one past the last element after casting
   *---------------------------------------------------------------------------**/
  template <typename T>
  std::enable_if_t<is_fixed_width<T>(), iterator<T>>
  end() {
    return iterator<T>{count_it{size()}, detail::mutable_value_accessor<T>{*this}};
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return column_view The requested child `column_view`
   *---------------------------------------------------------------------------**/
  __device__ mutable_column_device_view& child(size_type child_index) const
      noexcept {
    return mutable_children[child_index];
  }

  /**---------------------------------------------------------------------------*
   * @brief Updates the null mask to indicate that the specified element is
   * valid
   *
   * @note This operation requires a global atomic operation. Therefore, it is
   * not reccomended to use this function in performance critical regions. When
   * possible, it is more efficient to compute and update an entire word at
   * once using `set_mask_word`.
   * 
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   *---------------------------------------------------------------------------**/
  __device__ void set_valid(size_type element_index) const noexcept {
    return set_bit(null_mask(), element_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Updates the null mask to indicate that the specified element is null
   *
   * @note This operation requires a global atomic operation. Therefore, it is
   * not reccomended to use this function in performance critical regions. When
   * possible, it is more efficient to compute and update an entire word at
   * once using `set_mask_word`.
   * 
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   *---------------------------------------------------------------------------**/
  __device__ void set_null(size_type element_index) const noexcept {
    return clear_bit(null_mask(), element_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Updates the specified bitmask word in the `null_mask()` with a
   * new word.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   * @param new_element The new bitmask element
   *---------------------------------------------------------------------------**/
  __device__ void set_mask_word(size_type word_index,
                                bitmask_type new_word) const noexcept {
    null_mask()[word_index] = new_word;
  }

  /**---------------------------------------------------------------------------*
   * @brief Return the size in bytes of the amount of memory needed to hold a
   * device view of the specified column and it's children.
   *
   * @param source_view The `column_view` to use for this calculation.
   *---------------------------------------------------------------------------**/
  static size_type extent(mutable_column_view source_view);

 private:
  mutable_column_device_view*
      mutable_children{};    ///< Array of `mutable_column_device_view`
                             ///< objects in device memory.
                             ///< Based on element type, children
                             ///< may contain additional data
  size_type num_children{};  ///< The number of child columns

  /**---------------------------------------------------------------------------*
   * @brief Construct's a `mutable_column_device_view` from a
   *`mutable_column_view` populating all but the children.
   *
   * @note This constructor is for internal use only. To create a
   *`mutable_column_device_view` from a `column_view`, the
   *`mutable_column_device_view::create()` function should be used.
   *---------------------------------------------------------------------------**/
  mutable_column_device_view(mutable_column_view source);

  /**---------------------------------------------------------------------------*
   * @brief Destroy the `device_column_view` object.
   *
   * @note Does not free the column data, simply free's the device memory
   * allocated to hold the child views.
   *---------------------------------------------------------------------------**/
  void destroy();
};

/**---------------------------------------------------------------------------*
 * @brief Returns `string_view` to the string element at the specified index.
 *
 * If the element at the specified index is NULL, i.e., `is_null(element_index)
 *== true`, then any attempt to use the result will lead to undefined behavior.
 *
 * This function accounts for the offset.
 *
 * @param element_index Position of the desired string element
 * @return string_view instance representing this element at this index
 *---------------------------------------------------------------------------**/
template <>
__device__ inline string_view const column_device_view::element<string_view>(
    size_type element_index) const noexcept {
  size_type index =
      element_index + offset();  // account for this view's _offset
  const int32_t* d_offsets =
      d_children[strings_column_view::offsets_column_index].data<int32_t>();
  const char* d_strings =
      d_children[strings_column_view::chars_column_index].data<char>();
  size_type offset = d_offsets[index];
  return string_view{d_strings + offset, d_offsets[index + 1] - offset};
}

namespace detail {
/** -------------------------------------------------------------------------*
 * @brief value accessor of column without null bitmask
 * A unary functor returns scalar value at `id`.
 * `operator() (cudf::size_type id)` computes `element`
 * This functor is only allowed for non-nullable columns.
 *
 * the return value for element `i` will return `column[i]`
 *
 * @throws `cudf::logic_error` if the column is nullable.
 * @throws `cudf::logic_error` if column datatype and template T type mismatch.
 *
 * @tparam T The type of elements in the column
 * -------------------------------------------------------------------------**/

template <typename T>
struct value_accessor {
  column_device_view const col;  ///< column view of column in device

  /** -------------------------------------------------------------------------*
   * @brief constructor
   * @param[in] _col column device view of cudf column
   * -------------------------------------------------------------------------**/
  value_accessor(column_device_view const& _col) : col{_col} {
    CUDF_EXPECTS(data_type(experimental::type_to_id<T>()) == col.type(),
                 "the data type mismatch");
    CUDF_EXPECTS(!_col.has_nulls(), "Unexpected column with nulls.");
  }

  __device__ T operator()(cudf::size_type i) const { return col.element<T>(i); }
};

/** -------------------------------------------------------------------------*
 * @brief pair accessor of column with/without null bitmask
 * A unary functor returns pair with scalar value at `id` and boolean validity
 * `operator() (cudf::size_type id)` computes `element`  and
 * returns a `pair(element, validity)`
 *
 * the return value for element `i` will return `pair(column[i], validity)`
 * `validity` is `true` if `has_nulls=false`.
 * `validity` is validity of the element at `i` if `has_nulls=true` and the
 * column is nullable.
 *
 * @throws `cudf::logic_error` if `has_nulls==true` and the column is not
 * nullable.
 * @throws `cudf::logic_error` if column datatype and template T type mismatch.
 *
 * @tparam T The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 * -------------------------------------------------------------------------**/
template <typename T, bool has_nulls = false>
struct pair_accessor {
  column_device_view const col; ///< column view of column in device

  /** -------------------------------------------------------------------------*
   * @brief constructor
   * @param[in] _col column device view of cudf column
   * -------------------------------------------------------------------------**/
  pair_accessor(column_device_view const &_col) : col{_col} {
    CUDF_EXPECTS(data_type(experimental::type_to_id<T>()) == col.type(),
                 "the data type mismatch");
    if (has_nulls) {
      CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column.");
    }
  }

  CUDA_DEVICE_CALLABLE
  thrust::pair<T, bool> operator()(cudf::size_type i) const {
    return {col.element<T>(i), (has_nulls ? col.is_valid_nocheck(i) : true)};
  }
};

template <typename T>
struct mutable_value_accessor {
  mutable_column_device_view col;  ///< mutable column view of column in device

  /** -------------------------------------------------------------------------*
   * @brief constructor
   * @param[in] _col mutable column device view of cudf column
   * -------------------------------------------------------------------------**/
  mutable_value_accessor(mutable_column_device_view& _col) : col{_col} {
    CUDF_EXPECTS(data_type(experimental::type_to_id<T>()) == col.type(),
                 "the data type mismatch");
  }

  __device__ T& operator()(cudf::size_type i) { return col.element<T>(i); }
};

}  // namespace detail
}  // namespace cudf
