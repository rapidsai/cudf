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

#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Returns the number of times that the target string appears
 * in the source string.
 *
 * If `start <= 0` the search begins at the beginning of the `source` string.
 * If `end <=0` or `end` is greater the length of the `source` string,
 * the search stops at the end of the string.
 *
 * @param source Source string to search
 * @param target String to match within source
 * @param start First character position within source to start the search
 * @param end Last character position (exclusive) within source to search
 * @return Number of matches
 */
__device__ inline cudf::size_type count(string_view const source,
                                        string_view const target,
                                        cudf::size_type start = 0,
                                        cudf::size_type end   = -1)
{
  auto const tgt_length = target.length();
  auto const src_length = source.length();

  start = start < 0 ? 0 : start;
  end   = (end < 0 || end > src_length) ? src_length : end;

  if (tgt_length == 0) { return (end - start) + 1; }
  cudf::size_type count = 0;
  cudf::size_type pos   = start;
  while (pos != cudf::string_view::npos) {
    pos = source.find(target, pos, end - pos);
    if (pos != cudf::string_view::npos) {
      ++count;
      pos += tgt_length;
    }
  }
  return count;
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
