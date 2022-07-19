/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <memory>
#include <string>
#include <vector>

// somewhat iffy forward reference
namespace jitify {
class Program;
}

/**
 * @brief Module instance returned by create_udf_module
 */
struct udf_module {
  jitify::Program *program{};
  udf_module(jitify::Program *p) : program(p) {}
  ~udf_module();
};

/**
 * @brief Build UDF module with kernel source and compile options.
 *
 * @param udf String containing kernel source code
 * @param options Compile options for the jitify compiler
 * @return udf module to pass to `call_udf`
 */
std::unique_ptr<udf_module> create_udf_module(std::string const &udf,
                                              std::vector<std::string> const &options);

/**
 * @brief Launch kernel named `udf_name` in UDF module.
 *
 * @param udf Returned from `create_udf_module`
 * @param udf_name Name of global device function to execute in UDF module
 * @param output_size Output strings column size.
 *                    Also used to computing the thread block size.
 * @param input libcudf columns to pass to the kernel
 * @param heap_size Size in bytes to reserve for device-side malloc.
 * @return New libcudf strings column created by the kernel logic.
 */
std::unique_ptr<cudf::column> call_udf(udf_module const &udf,
                                       std::string const &udf_name,
                                       cudf::size_type output_size,
                                       std::vector<cudf::column_view> input,
                                       size_t heap_size = 1073741824);

/**
 * @brief Return a cudf::string_view array for the given strings column
 *
 * @throw cudf::logic_error if input is not a strings column.
 */
std::unique_ptr<rmm::device_buffer> to_string_view_array(cudf::column_view const input);

/**
 * @brief Return a cudf::column given an array of dstring objects.
 *
 * @param d_buffer Pointer to device memory of dstring objects
 * @param d_size The number of bytes in the d_buffer
 * @return A strings column copy of the dstring objects
 */
std::unique_ptr<cudf::column> from_dstring_array(void *d_buffer, std::size_t size);

