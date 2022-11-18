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
#include <cudf/strings/udf/udf_string.cuh>

namespace cudf {
namespace strings {
namespace udf {

/**
 * @brief Returns new string replacing all occurrences of target with replacement
 *
 * @param source Source string to search
 * @param target String to match within source
 * @param replacement String to replace the target within the source
 * @return Resulting string
 */
__device__ inline udf_string replace(string_view source,
                                     string_view target,
                                     string_view replacement)
{
  if (target.empty()) return udf_string(source);

  udf_string result;

  size_type last_position = 0;
  auto position           = source.find(target, 0);
  while (position != string_view::npos) {
    auto left = source.substr(last_position, position - last_position);
    result.append(left);
    result.append(replacement);
    last_position = position + target.length();
    position      = source.find(target, last_position);
  }
  if (last_position < source.length()) {
    auto right = source.substr(last_position, source.length() - last_position);
    result.append(right);
  }

  return result;
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
