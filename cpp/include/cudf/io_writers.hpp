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

#include <memory>
#include <string>
#include <vector>

#include <cudf/legacy/table.hpp>
#include "cudf.h"

namespace cudf {
namespace io {
namespace orc {

/**---------------------------------------------------------------------------*
 * @brief Supported compression algorithms for the ORC writer
 *---------------------------------------------------------------------------**/
enum class compression_type { none, snappy };

/**---------------------------------------------------------------------------*
 * @brief Options for the ORC writer
 *---------------------------------------------------------------------------**/
struct writer_options {
  compression_type compression = compression_type::none;

  writer_options() = default;
  writer_options(writer_options const &) = default;

  /**---------------------------------------------------------------------------*
   * @brief Constructor to populate writer options.
   *
   * @param[in] comp Compression codec to use
   *---------------------------------------------------------------------------**/
  explicit writer_options(compression_type comp) : compression(comp) {}
};

/**---------------------------------------------------------------------------*
 * @brief Class to write ORC data into cuDF columns
 *---------------------------------------------------------------------------**/
class writer {
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

 public:
  /**---------------------------------------------------------------------------*
   * @brief Constructor for a file path source.
   *---------------------------------------------------------------------------**/
  explicit writer(std::string filepath, writer_options const &options);

  /**---------------------------------------------------------------------------*
   * @brief Writes the entire data set.
   *
   * @param[in] table Object that contains the array of gdf_columns
   *---------------------------------------------------------------------------**/
  void write_all(const cudf::table& table);

  ~writer();
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
