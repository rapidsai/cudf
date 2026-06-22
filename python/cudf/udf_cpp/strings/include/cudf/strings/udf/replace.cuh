/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
 * If target is empty then replacement is inserted between every character.
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
  udf_string result;
  auto const tgt_length   = target.length();
  auto const src_length   = source.length();
  size_type last_position = 0;
  size_type position      = 0;
  while (position != string_view::npos) {
    position = source.find(target, last_position);
    if (position != string_view::npos) {
      result.append(source.substr(last_position, position - last_position));
      result.append(replacement);
      last_position = position + tgt_length;
      if ((tgt_length == 0) && (++last_position <= src_length)) {
        result.append(source.substr(position, 1));
      }
    }
  }
  if (last_position < src_length) {
    result.append(source.substr(last_position, src_length - last_position));
  }

  return result;
}

}  // namespace udf
}  // namespace strings
}  // namespace cudf
