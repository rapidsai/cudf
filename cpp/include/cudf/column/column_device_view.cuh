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
#include <cudf/types.hpp>

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief An immutable, non-owning view of device data as a column of elements
 * that is trivially copyable into CUDA device code.
 *---------------------------------------------------------------------------**/
class column_device_view {
 public:
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
  static auto create(column_view source_view, cudaStream_t stream = 0);

  column_device_view() = delete;
  column_device_view(column_device_view const&) = default;
  column_device_view(column_device_view&&) = default;
  column_device_view& operator=(column_device_view const&) = default;
  column_device_view& operator=(column_device_view&&) = default;

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
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return column_view The requested child `column_view`
   *---------------------------------------------------------------------------**/
  __device__ column_device_view child(size_type child_index) const noexcept {
    assert(child_index > 0);
    return d_children[child_index];
  }

  __device__ bool is_valid(size_type element_index) {
    // TODO Implement
    return true;
  }

  __device__ bool is_null(size_type element_index) {
    return not is_valid(element_index);
  }

 private:
  data_type _type{INVALID};  ///< Element type
  cudf::size_type _size{};   ///< Number of elements
  void const* _data{};       ///< Pointer to device memory containing elements
  bitmask_type const* _null_mask{};  ///< Pointer to device memory containing
                                     ///< bitmask representing null elements.
                                     ///< Optional if `null_count() == 0`
  size_type _null_count{};           ///< The number of null elements
  size_type _offset{};               ///< Index position of the first element.
                                     ///< Enables zero-copy slicing
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

}  // namespace cudf