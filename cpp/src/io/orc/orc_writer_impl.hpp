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
#include <fstream>

#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {
namespace io {
namespace orc {

/**
 * @brief Implementation for ORC writer
 **/
class writer::Impl {
 public:
  /**
   * @brief Constructor with writer options.
   **/
  explicit Impl(std::string filepath, writer_options const &options);

  /**
   * @brief Write an entire dataset to ORC format
   * 
   * @param[in] table Table of columns
   **/
  void write(const cudf::table& table);

 private:
  void write_metadata();

  void write_postscript();

  void write_footer();

  void write_stripes();

 private:
  std::ofstream outfile_;
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
