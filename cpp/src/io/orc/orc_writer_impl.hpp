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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {
namespace io {
namespace orc {

// Forward internal classes
class ProtobufWriter;

/**
 * @brief Implementation for ORC writer
 **/
class writer::Impl {
  static constexpr const char* magic = "ORC";

 public:
  /**
   * @brief Constructor with writer options.
   **/
  explicit Impl(std::string filepath, writer_options const& options);

  /**
   * @brief Write an entire dataset to ORC format
   *
   * @param[in] table Table of columns
   **/
  void write(const cudf::table& table);

 private:
  size_t write_stripes();

  size_t write_filefooter(const cudf::table& table);

  size_t write_postscript(size_t ff_length);

 private:
  std::ofstream outfile_;
  std::vector<uint8_t> filetail_buffer_;
  std::unique_ptr<ProtobufWriter> pbw_;
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
