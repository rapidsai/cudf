/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef SCALAR_WRAPPER_H
#define SCALAR_WRAPPER_H

#include <cudf/cudf.h>
#include "cudf_test_utils.cuh"
#include <rmm/rmm.h>
#include <utilities/bit_util.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <thrust/equal.h>
#include <thrust/logical.h>
#include <bitset>

namespace cudf {
namespace test {

/**---------------------------------------------------------------------------*
 * @brief Wrapper for a gdf_scalar used for unit testing.
 *
 * An abstraction on top of a gdf_scalar that provides functionality for
 * allocating, intiailizing, and otherwise managing gdf_scalars for passing to
 * libcudf APIs in unit testing.
 *
 * @tparam ColumnType The underlying data type of the scalar
 *---------------------------------------------------------------------------**/
template <typename ScalarType>
struct scalar_wrapper {
  /**---------------------------------------------------------------------------*
   * @brief Implicit conversion operator to a gdf_scalar pointer.
   *
   * In this way, a column_wrapper can be passed directly into a libcudf API
   * and will be implicitly converted to a pointer to its underlying gdf_scalar
   * without the need to use the `get()` member.
   *
   * @return gdf_scalar* Pointer to the underlying gdf_scalar
   *---------------------------------------------------------------------------**/
  operator gdf_scalar*() { return &the_scalar; };

  operator gdf_scalar&() { return the_scalar; };


  /**---------------------------------------------------------------------------*
   * @brief Construct a new scalar wrapper object
   *
   * Constructs a scalar_wrapper using a value of type ScalarType.
   *
   * @param value The value to use for the scalar
   * @param is_valid is the scalar valid
   *---------------------------------------------------------------------------**/
  scalar_wrapper(ScalarType value, bool is_valid = true) {
    auto dataptr = reinterpret_cast<ScalarType*>(&(the_scalar.data));
    *dataptr = value;
    the_scalar.is_valid = is_valid;
    the_scalar.dtype = gdf_dtype_of<ScalarType>();
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new scalar wrapper object
   *
   * Constructs a scalar_wrapper using a gdf_scalar
   *
   * @param scalar The scalar value to hold
   *---------------------------------------------------------------------------**/
  scalar_wrapper(const gdf_scalar& scalar) : the_scalar(scalar) {}

  /**---------------------------------------------------------------------------*
   * @brief Returns a pointer to the underlying gdf_scalar.
   *---------------------------------------------------------------------------**/
  gdf_scalar* get() { return &the_scalar; }
  gdf_scalar const* get() const { return &the_scalar; }

  /**---------------------------------------------------------------------------*
   * @brief returns the value of the scalar
   *
   * @return ScalarType
   *---------------------------------------------------------------------------**/
  ScalarType value() const {
    return *reinterpret_cast<ScalarType const*>(&(the_scalar.data));
  }

  /**---------------------------------------------------------------------------*
   * @brief returns the validity of the scalar
   *
   * @return bool
   *---------------------------------------------------------------------------**/
  bool is_valid() const { return the_scalar.is_valid; }

  /**---------------------------------------------------------------------------*
   * @brief Prints the value of the underlying gdf_scalar.
   *---------------------------------------------------------------------------**/
  void print() const {
    ScalarType value = *reinterpret_cast<ScalarType const*>(&(the_scalar.data));
    if (the_scalar.is_valid)
      std::cout << value << std::endl;
    else
      std::cout << "null" << std::endl;
  }

  /**---------------------------------------------------------------------------*
   * @brief CompCompares this wrapper with another scalar_wrapper for equality.
   *
   * @param rhs  The other scalar_wrapper to check for equality
   * @return true The two scalars are equal
   * @return false The two scalars are not equal
   *---------------------------------------------------------------------------**/
  bool operator==(scalar_wrapper<ScalarType> const& rhs) const {
    return (the_scalar.dtype == rhs.the_scalar.dtype and
            this->is_valid() == rhs.is_valid() and
            this->value() == rhs.value());
  }

 private:
  gdf_scalar the_scalar;
};

}  // namespace test
}  // namespace cudf
#endif
