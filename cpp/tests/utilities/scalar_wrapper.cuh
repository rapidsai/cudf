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

#include "cudf.h"
#include "cudf_test_utils.cuh"
#include "rmm/rmm.h"
#include "utilities/bit_util.cuh"
#include "utilities/type_dispatcher.hpp"

#include <thrust/equal.h>
#include <thrust/logical.h>
#include <bitset>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  do {                                                                        \
    cudaError_t cudaStatus = (call);                                          \
    if (cudaSuccess != cudaStatus) {                                          \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
      exit(1);                                                                \
    }                                                                         \
  } while (0)
#endif

namespace cudf {
namespace test {

/**---------------------------------------------------------------------------*
 * @brief Wrapper for a gdf_scalar used for unit testing.
 *
 * An abstraction on top of a gdf_scalar that provides functionality for
 * allocating, intiailizing, and otherwise managing gdf_scalar's for passing to
 * libcudf APIs in unit testing.
 *
 * @tparam ColumnType The underlying data type of the scalar
 *---------------------------------------------------------------------------**/
template <typename ScalarType>
struct scalar_wrapper {
  /**---------------------------------------------------------------------------*
   * @brief Implicit conversion operator to a gdf_scalar pointer.
   *
   * Allows for implicit conversion of a column_wrapper to a pointer to its
   * underlying gdf_scalar.
   *
   * In this way, a column_wrapper can be passed directly into a libcudf API
   * and will be implicitly converted to a pointer to its underlying gdf_scalar
   * without the need to use the `get()` member.
   *
   * @return gdf_scalar* Pointer to the underlying gdf_scalar
   *---------------------------------------------------------------------------**/
  operator gdf_scalar*(){return &the_scalar;};

  /**---------------------------------------------------------------------------*
   * @brief Construct a new scalar wrapper object
   *
   * Constructs a scalar_wrapper using a ref value for the host data.
   *
   * @param host_data The value to use for the scalar
   *---------------------------------------------------------------------------**/
  scalar_wrapper(ScalarType const& host_data) {
    initialize_with_host_data(host_data);
  }

  ~scalar_wrapper() {
    RMM_FREE(the_scalar.data, 0);
  }

  /**---------------------------------------------------------------------------*
   * @brief Returns a pointer to the underlying gdf_scalar.
   *
   *---------------------------------------------------------------------------**/
  gdf_scalar* get() { return &the_scalar; }
  gdf_scalar const* get() const { return &the_scalar; }

  /**---------------------------------------------------------------------------*
   * @brief Copies the underying gdf_scalar's data to the host.
   *
   * Returns the value of the scalar
   *
   *---------------------------------------------------------------------------**/
  auto to_host() const {
    ScalarType host_data;

    if (nullptr != the_scalar.data) {
      CUDA_RT_CALL(cudaMemcpy(&host_data, the_scalar.data,
                              sizeof(ScalarType),
                              cudaMemcpyDeviceToHost));
    }

    return host_data;
  }

  /**---------------------------------------------------------------------------*
   * @brief Prints the value of the underlying gdf_scalar.
   *
   *---------------------------------------------------------------------------**/
  void print() const {
    ScalarType value = this->to_host();
    std::cout << value << std::endl;
  }

  /**---------------------------------------------------------------------------*
   * @brief Compares if another scalar_wrapper is equal to this wrapper.
   *
   * @param rhs  The other scalar_wrapper to check for equality
   * @return true The two scalars are equal
   * @return false The two scalars are not equal
   *---------------------------------------------------------------------------**/
  bool operator==(scalar_wrapper<ScalarType> const& rhs) const {
    if (the_scalar.dtype != rhs.the_scalar.dtype) return false;

    if (!(the_scalar.data && rhs.the_scalar.data))
      return false;  // if one is null but not both

    return (this->to_host() == rhs.to_host());
  }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates and initializes the underyling gdf_scalar with host data.
   *
   * Creates a gdf_scalar and copies data from the host for it's data. 
   * Sets the corresponding dtype based on the scalar_wrapper's ColumnType.
   *
   * @param host_data The host data to copy to device for the scalar's data
   *---------------------------------------------------------------------------**/
  void initialize_with_host_data(ScalarType const& host_data) {
    // Allocate device storage for gdf_scalar and copy contents from host_data
    RMM_ALLOC(&(the_scalar.data), sizeof(ScalarType), 0);
    CUDA_RT_CALL(cudaMemcpy(the_scalar.data, &host_data,
                            sizeof(ScalarType),
                            cudaMemcpyHostToDevice));

    // Fill the gdf_scalar members
    the_scalar.dtype = cudf::type_to_gdf_dtype<ScalarType>::value;
  }

  gdf_scalar the_scalar;
};

}  // namespace test
}  // namespace cudf
#endif
