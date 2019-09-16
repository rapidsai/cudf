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

#include <cuda_runtime.h>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.cuh>
#include <cudf/strings/string_view.cuh>

namespace cudf {

namespace detail {

/**---------------------------------------------------------------------------*
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable CUDA device code.
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
   *a column, and instead, accessing the elements should be done via
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
   * @brief Indicates if the column can contain null elements, i.e., if it has
   * an allocated bitmask.
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
   * @brief Indicates if the column contains null elements,
   * i.e., `null_count() > 0`
   *
   * @return true The column contains null elements
   * @return false All elements are valid
   *---------------------------------------------------------------------------**/
  __host__ __device__ bool has_nulls() const noexcept {
    return _null_count > 0;
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
   * @brief Returns if the specified element holds a valid value (i.e., not
   * null)
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to query
   * @return true The element is valid
   * @return false The element is null
   *---------------------------------------------------------------------------**/
  __device__ bool is_valid(size_type element_index) const noexcept {
    return bit_is_set(_null_mask, element_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns if the specified element is null
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to query
   * @return true The element is null
   * @return false The element is valid
   *---------------------------------------------------------------------------**/
  __device__ bool is_null(size_type element_index) const noexcept {
    return not is_valid(element_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns the the specified bitmask element from the `null_mask()`.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index
   * @return __device__ get_mask_element
   *---------------------------------------------------------------------------**/
  __device__ bitmask_type get_mask_element(size_type element_index) const noexcept {
    return null_mask()[element_index];
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
}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable CUDA device code.
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
   * @brief Factory to construct a column view that is usable in device memory.
   *
   * Allocates and copies views of `soure_view`'s children to device memory to
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
  static std::unique_ptr<column_device_view, std::function<void(column_device_view*)>> create(column_view source_view, cudaStream_t stream = 0);
  
  /**---------------------------------------------------------------------------*
   * @brief Returns reference to element at the specified index.
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

  /**---------------------------------------------------------------------------*
   * @brief Destroy the `device_column_view` object.
   *
   * @note Does not free the column data, simply free's the device memory
   * allocated to hold the child views.
   *---------------------------------------------------------------------------**/
  void destroy();
};

/**---------------------------------------------------------------------------*
 * @brief A mutable, non-owning view of device data as a column of elements
 * that is trivially copyable and usable CUDA device code.
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
  static auto create(mutable_column_view source_view, cudaStream_t stream = 0);

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
   * @return T const* Typed pointer to underlying data
   *---------------------------------------------------------------------------**/
  template <typename T = void>
  __host__ __device__ T* head() const noexcept {
    return const_cast<T*>(detail::column_device_view_base::head());
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
  __host__ __device__ T* data() const noexcept {
    return head<T>() + _offset;
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
  __device__ T element(size_type element_index) noexcept {
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
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   *---------------------------------------------------------------------------**/
  __device__ void set_null(size_type element_index) const noexcept {
    return clear_bit(null_mask(), element_index);
  }

  /**---------------------------------------------------------------------------*
   * @brief Updates the specified bitmask element in the `null_mask()` with a
   * new element.
   *
   * @note It is undefined behavior to call this function if `nullable() ==
   * false`.
   *
   * @param element_index The index of the element to update
   * @param new_element The new bitmask element
   *---------------------------------------------------------------------------**/
  __device__ void set_mask_element(size_type element_index,
                              bitmask_type new_element) const noexcept {
    null_mask()[element_index] = new_element;
  }

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
   * This function accounts for the offset.
   *
   * @param element_index Position of the desired string
   *---------------------------------------------------------------------------**/
  
  template <>
  __device__ inline string_view const column_device_view::element<string_view>(
      size_type element_index) const noexcept {
        size_type index = element_index + _offset; // account for this view's _offset
        const int32_t* d_offsets = d_children[0].data<int32_t>();
        const char* d_strings = d_children[1].data<char>();
        size_type offset = index ? d_offsets[index-1] : 0;
        return string_view{d_strings + offset, d_offsets[index] - offset};
  }

  //template <>
  //__device__ inline string_view mutable_column_device_view::element<string_view>(
  //    size_type element_index) noexcept {
  //      return string_view{};
  //}


}  // namespace cudf