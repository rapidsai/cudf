/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace cudf {
namespace io {

/**
 * @brief TODO
 **/
class data_sink {
 public:
  /**
   * @brief Create a sink from a file path
   *
   * @param[in] filepath Path to the file to use
   **/
  static std::unique_ptr<data_sink> create(const std::string& filepath);

  /**
   * @brief Base class destructor
   **/
  virtual ~data_sink(){};

  /**
   * @brief TODO
   *
   * @param[in] data TODO
   * @param[in] size Bytes to write
   *
   * @return void
   **/
  virtual void write(void const* data, size_t size) = 0;

  /**
   * @brief TODO
   * 
   * @return void
   */
  virtual void flush() = 0;
  
  /**
   * @brief TODO
   *
   * @return size_t TODO
   **/
  virtual size_t position() = 0;
};

}  // namespace io
}  // namespace cudf
