/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

__device__ void print_debug(cudf::strings::udf::dstring const& d_str, char const* name = "")
{
  printf(
    "%s:(%p,%d,%d)=[%s]\n", name, d_str.data(), d_str.size_bytes(), d_str.capacity(), d_str.data());
}

__device__ void verify(cudf::strings::udf::dstring const& d_str,
                       cudf::string_view const expected,
                       char const* name = 0)
{
  if (d_str.compare(expected) == 0) {
    printf("\x1B[32mOK\x1B[0m: %s\n", name);
  } else {
    auto exp_str = cudf::strings::udf::dstring(expected);
    printf("\x1B[31mError\x1B[0m: %s [%s]!=[%s]\n", name, d_str.data(), exp_str.data());
  }
}

__device__ void check_result(bool result, char const* str = 0)
{
  if (result) {
    printf("\x1B[32mOK\x1B[0m: %s\n", str);
  } else {
    printf("\x1B[31mError\x1B[0m: %s\n", str);
  }
}
