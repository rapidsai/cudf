/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/type_traits>

namespace cudf {
namespace jit {
namespace detail {

/// @brief A minified version of `cudf::detail::column_device_view_base` for use in JIT kernels.
class alignas(16) column_device_view_base {
 public:
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

  /**
   * @copydoc cudf::detail::column_device_view_base::head
   */
  template <typename T = void,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  [[nodiscard]] CUDF_HOST_DEVICE T const* head() const noexcept
  {
    return static_cast<T const*>(_data);
  }

  /**
   * @copydoc cudf::detail::column_device_view_base::data
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] CUDF_HOST_DEVICE T const* data() const noexcept
  {
    return head<T>() + _offset;
  }

  /**
   * @copydoc cudf::detail::column_device_view_base::size
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type size() const noexcept { return _size; }

  /**
   * @copydoc cudf::detail::column_device_view_base::type
   */
  [[nodiscard]] CUDF_HOST_DEVICE data_type type() const noexcept { return _type; }

  /**
   * @copydoc cudf::detail::column_device_view_base::nullable
   */
  [[nodiscard]] CUDF_HOST_DEVICE bool nullable() const noexcept { return nullptr != _null_mask; }

  /**
   * @copydoc cudf::detail::column_device_view_base::null_mask
   */
  [[nodiscard]] CUDF_HOST_DEVICE bitmask_type const* null_mask() const noexcept
  {
    return _null_mask;
  }

  /**
   * @copydoc cudf::detail::column_device_view_base::offset
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type offset() const noexcept { return _offset; }

  /**
   * @copydoc cudf::detail::column_device_view_base::is_valid
   */
  [[nodiscard]] __device__ bool is_valid(size_type element_index) const noexcept
  {
    return not nullable() or is_valid_nocheck(element_index);
  }

  /**
   * @copydoc cudf::detail::column_device_view_base::is_valid_nocheck
   */
  [[nodiscard]] __device__ bool is_valid_nocheck(size_type element_index) const noexcept
  {
    return bit_is_set(_null_mask, offset() + element_index);
  }

  /**
   * @copydoc cudf::detail::column_device_view_base::is_null
   */
  [[nodiscard]] __device__ bool is_null(size_type element_index) const noexcept
  {
    return not is_valid(element_index);
  }

  /**
   * @copydoc cudf::detail::column_device_view_base::is_null_nocheck
   */
  [[nodiscard]] __device__ bool is_null_nocheck(size_type element_index) const noexcept
  {
    return not is_valid_nocheck(element_index);
  }

  /**
   * @copydoc cudf::detail::column_device_view_base::get_mask_word
   */
  [[nodiscard]] __device__ bitmask_type get_mask_word(size_type word_index) const noexcept
  {
    return null_mask()[word_index];
  }

 protected:
  /**
   * @copydoc cudf::detail::column_device_view_base::_type
   */
  data_type _type;

  /**
   * @copydoc cudf::detail::column_device_view_base::_size
   */
  size_type _size;

  /**
   * @copydoc cudf::detail::column_device_view_base::_data
   */
  void const* _data;

  /**
   * @copydoc cudf::detail::column_device_view_base::_null_mask
   */
  bitmask_type const* _null_mask;

  /**
   * @copydoc cudf::detail::column_device_view_base::_offset
   */
  size_type _offset;
};

}  // namespace detail

/// @brief A minified version of `cudf::column_device_view` for use in JIT kernels.
class alignas(16) column_device_view : public detail::column_device_view_base {
 public:
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

  /**
   * @copydoc cudf::column_device_view::element
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    return data<T>()[element_index];
  }

  /**
   * @copydoc cudf::column_device_view::element
   */
  template <typename T, CUDF_ENABLE_IF(is_fixed_point<T>())>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    using namespace numeric;
    using rep        = typename T::rep;
    auto const scale = scale_type{_type.scale()};
    return T{scaled_integer<rep>{data<rep>()[element_index], scale}};
  }

  /**
   * @copydoc cudf::column_device_view::child
   */
  [[nodiscard]] __device__ column_device_view child(size_type child_index) const noexcept
  {
    return d_children[child_index];
  }

  /**
   * @copydoc cudf::column_device_view::num_child_columns
   */
  [[nodiscard]] CUDF_HOST_DEVICE size_type num_child_columns() const noexcept
  {
    return _num_children;
  }

 protected:
  /**
   * @copydoc cudf::column_device_view::d_children
   */
  column_device_view* d_children = nullptr;

  /**
   * @copydoc cudf::column_device_view::_num_children
   */
  size_type _num_children = 0;
};

/// @brief A minified version of `cudf::mutable_column_device_view` for use in JIT kernels.
class alignas(16) mutable_column_device_view : public detail::column_device_view_base {
 public:
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
  mutable_column_device_view(data_type type,
                             size_type size,
                             void* data,
                             bitmask_type const* null_mask,
                             size_type offset,
                             mutable_column_device_view* children,
                             size_type num_children)
    : column_device_view_base(type, size, data, null_mask, offset),
      d_children(children),
      _num_children(num_children)
  {
  }

  /**
   * @copydoc cudf::mutable_column_device_view::head
   */
  template <typename T = void,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, void> or is_rep_layout_compatible<T>())>
  CUDF_HOST_DEVICE T* head() const noexcept
  {
    return const_cast<T*>(detail::column_device_view_base::head<T>());
  }

  /**
   * @copydoc cudf::mutable_column_device_view::data
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  CUDF_HOST_DEVICE T* data() const noexcept
  {
    return const_cast<T*>(detail::column_device_view_base::data<T>());
  }

  /**
   * @copydoc cudf::mutable_column_device_view::element
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  [[nodiscard]] __device__ T& element(size_type element_index) const noexcept
  {
    return data<T>()[element_index];
  }

  /**
   * @copydoc cudf::mutable_column_device_view::element
   */
  template <typename T, CUDF_ENABLE_IF(is_fixed_point<T>())>
  [[nodiscard]] __device__ T element(size_type element_index) const noexcept
  {
    using namespace numeric;
    using rep        = typename T::rep;
    auto const scale = scale_type{_type.scale()};
    return T{scaled_integer<rep>{data<rep>()[element_index], scale}};
  }

  /**
   * @brief Assigns `value` to the element at `element_index`
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @param value The value to assign
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  __device__ void assign(size_type element_index, T value) const noexcept
  {
    data<T>()[element_index] = value;
  }

  /**
   * @brief Assigns `value` to the element at `element_index`.
   * @warning Expects that `value` has been scaled to the column's scale
   *
   * @tparam T The element type
   * @param element_index Position of the desired element
   * @param value The value to assign
   */
  template <typename T, CUDF_ENABLE_IF(is_fixed_point<T>())>
  __device__ void assign(size_type element_index, T value) const noexcept
  {
    // consider asserting that the scale matches
    using namespace numeric;
    using rep                  = typename T::rep;
    data<rep>()[element_index] = value.value();
  }

  /**
   * @copydoc cudf::mutable_column_device_view::child
   */
  [[nodiscard]] __device__ mutable_column_device_view child(size_type child_index) const noexcept
  {
    return d_children[child_index];
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
  mutable_column_device_view* d_children;  ///< Array of `mutable_column_device_view`
                                           ///< objects in device memory.
                                           ///< Based on element type, children
                                           ///< may contain additional data
  size_type _num_children;                 ///< The number of child columns
};

}  // namespace jit
}  // namespace cudf
