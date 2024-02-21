
/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "udf_string.cuh"

#include <cudf/strings/detail/convert/int_to_string.cuh>
#include <cudf/strings/detail/convert/string_to_float.cuh>
#include <cudf/strings/detail/convert/string_to_int.cuh>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Converts a string into an integer
 *
 * The '+' and '-' are allowed but only at the beginning of the string.
 * The string is expected to contain base-10 [0-9] characters only.
 * Any other character will end the parse.
 * Overflow of the int64 type is not detected.
 */
__device__ inline int64_t stoi(string_view const& d_str)
{
  return cudf::strings::detail::string_to_integer(d_str);
}

/**
 * @brief Converts an integer into string
 *
 * @param value integer value to convert
 */
__device__ inline udf_string to_string(int64_t value)
{
  udf_string result;
  if (value == 0) {
    result.append("0");
    return result;
  }
  result.resize(cudf::strings::detail::count_digits(value));
  cudf::strings::detail::integer_to_string(value, result.data());
  return result;
}

/**
 * @brief Converts a string into a double
 *
 * This function supports scientific notation.
 * Overflow goes to inf or -inf and underflow may go to 0.
 */
__device__ inline double stod(string_view const& d_str)
{
  return cudf::strings::detail::stod(d_str);
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
