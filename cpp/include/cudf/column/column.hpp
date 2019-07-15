/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#include "column_view.hpp"
#include "mutable_column_view.hpp"

#include <rmm/device_buffer.hpp>

namespace cudf {

class column {
 public:
  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a size, type, and option to
   * allocate bitmask.
   *
   * Both the data and bitmask are unintialized.
   *
   * @param[in] type The element type
   * @param[in] size The number of elements in the column
   * @param[in] allocate_bitmask Optionally allocate an appropriate sized
   * bitmask
   *---------------------------------------------------------------------------**/
  column(data_type type, size_type size, bool allocate_bitmask = false);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and a device_buffer for
   * data that will be *deep* copied.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the column
   * @param[in] data device_buffer whose data will be *deep* copied
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer data);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and a device_buffer for
   * data that will be shallow copied.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the column
   * @param[in] data device_buffer whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer&& data);

  /**---------------------------------------------------------------------------*
   * @brief Column constructor that deep copies a `device_buffer` and `bitmask`.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements
   * @param[in] data The device buffer to copy
   * @param[in] mask The bitmask to copy
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer data, column mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and device_buffers for data and
   * bitmask that will be *shallow* copied.
   *
   * This constructor uses move semantics to take ownership of the
   *device_buffer's device memory. The `device_buffer` passed into this
   *constructor will not longer be valid to use. Furthermore, it will result in
   *undefined behavior if the device_buffer`s associated memory is modified or
   *freed after invoking this constructor.
   *
   * @param dtype The element type
   * @param[in] size The number of elements in the column
   * @param data device_buffer whose data will be moved from into this column
   * @param mask bitmask whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer&& data,
         column&& mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, size, and deep copied device
   * buffer for data, and moved bitmask.
   *
   * @param dtype The element type
   * @param size The number of elements
   * @param data device_buffer whose data will be *deep* copied
   * @param mask bitmask whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer data,
         column&& mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, size, and moved device
   * buffer for data, and deep copied bitmask.
   *
   * @param dtype The element type
   * @param size The number of elements
   * @param data device_buffer whose data will be moved into this column
   * @param mask bitmask whose data will be deep copied into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, size_type size, rmm::device_buffer&& data,
         column mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column by deep copying the device memory of another
   * column.
   *
   * @param other The other column to copy
   *---------------------------------------------------------------------------**/
  column(column const& other);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column object by moving the device memory from
   *another column.
   *
   * @param other The other column whose device memory will be moved to the new
   * column
   *---------------------------------------------------------------------------**/
  column(column&& other);

  ~column() = default;
  column& operator=(column const& other) = delete;
  column& operator=(column&& other) = delete;

  column_view view() const noexcept { return this->operator column_view(); }

  operator column_view() const;

  mutable_column_view mutable_view() noexcept {
    return this->operator mutable_column_view();
  }

  operator mutable_column_view();

 private:
  rmm::device_buffer _data{};  ///< Dense, contiguous, type erased device memory
                               ///< buffer containing the column elements
  std::unique_ptr<column>
      _null_mask{};         ///< Column of BOOL1 elements
                            ///< where `true` indicates an element
                            ///< is valid, `false` indicates "null". Optional if
                            ///< `null_count() == 0`
  size_type _null_count{};  ///< The number of null elements
  cudf::size_type _size{};  ///< The number of elements in the column
  data_type _type{INVALID};         ///< Logical type of elements in the column
  std::vector<column> _children{};  ///< Depending on element type, child
                                    ///< columns may contain additional data
  std::unique_ptr<column> dictionary{};
};
}  // namespace cudf