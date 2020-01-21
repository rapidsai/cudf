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

#include "data_sink.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cudf/cudf.h>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace io {

/**
 * @brief TODO
 * 
 */
class file_sink : public data_sink {

 public:
  explicit file_sink(const std::string& filepath): filepath_(filepath){

  }

  virtual ~file_sink() {
  }

  void write(void* data, size_t size) override {
  }

  size_t get_position() const override { return 0; }

 private:
  

 private:
  std::string filepath_;
};

std::unique_ptr<data_sink> data_sink::create(const std::string& filepath) {
  return std::make_unique<file_sink>(filepath.c_str());
}

}  // namespace io
}  // namespace cudf
