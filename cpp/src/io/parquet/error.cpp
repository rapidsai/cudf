/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "error.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_scalar.hpp>

#include <sstream>

namespace cudf::io::parquet {

namespace {

rmm::device_scalar<int32_t> error_code(0, cudf::get_default_stream());

}  // anonymous namespace

int32_t* get_error() { return error_code.data(); }

int32_t get_error_code() { return error_code.value(error_code.stream()); }

std::string get_error_string()
{
  std::stringstream sstream;
  sstream << std::hex << get_error_code();
  return " 0x" + sstream.str();
}

void reset_error_code() { error_code.set_value_to_zero_async(error_code.stream()); }

void set_error_stream(rmm::cuda_stream_view stream) { error_code.set_stream(stream); }

}  // namespace cudf::io::parquet
