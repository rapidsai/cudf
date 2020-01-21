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

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cudf/utilities/legacy/wrapper_types.hpp>
#include <cudf/cudf.h>

// Forward declarations
class NVStrings;
class NVCategory;

/**---------------------------------------------------------------------------*
 * @brief Checks if a file exists.
 *---------------------------------------------------------------------------**/
bool checkFile(std::string const &fname);

/**---------------------------------------------------------------------------*
 * @brief Check a string gdf_column, including the type and values.
 *---------------------------------------------------------------------------**/
void checkStrColumn(gdf_column const *col, std::vector<std::string> const &refs);

/**
 * @brief Generates a vector of uniform random values of type T
 **/
template <typename T>
inline auto random_values(size_t size) {
  std::vector<T> values(size);

  using T1 = cudf::detail::unwrapped_type_t<T>;
  using uniform_distribution = typename std::conditional_t<
      std::is_same<T1, bool>::value, std::bernoulli_distribution,
      std::conditional_t<std::is_floating_point<T1>::value,
                         std::uniform_real_distribution<T1>,
                         std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

/**---------------------------------------------------------------------------*
 * @brief Simple test internal helper class to transfer cudf column data
 * from device to host for test comparisons and debugging/development.
 *---------------------------------------------------------------------------**/
template <typename T> class gdf_host_column {
public:
  gdf_host_column() = delete;
  explicit gdf_host_column(const gdf_column *col) {
    m_hostdata = std::vector<T>(col->size);
    cudaMemcpy(m_hostdata.data(), col->data, sizeof(T) * col->size, cudaMemcpyDeviceToHost);
  }

  auto hostdata() const -> const auto & { return m_hostdata; }
  void print() const {
    for (size_t i = 0; i < m_hostdata.size(); ++i) {
      std::cout.precision(17);
      std::cout << "[" << i << "]: value=" << m_hostdata[i] << "\n";
    }
  }

private:
  std::vector<T> m_hostdata;
};

/**
 * @brief Returns list of strings from an NVString instance
 **/
std::vector<std::string> nvstrings_to_strings(NVStrings *nvstr);

/**
 * @brief Returns list of strings from an NVCategory instance
 **/
std::vector<std::string> nvcategory_to_strings(NVCategory *nvcat);
