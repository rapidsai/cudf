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

#include <io/utilities/row_selection.hpp>

#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <limits>

namespace cudf::io::detail {

std::pair<uint64_t, size_type> skip_rows_num_rows_from_options(
  uint64_t skip_rows_opt, std::optional<size_type> const& num_rows_opt, uint64_t num_source_rows)
{
  auto const rows_to_skip = std::min(skip_rows_opt, num_source_rows);
  if (not num_rows_opt.has_value()) {
    CUDF_EXPECTS(num_source_rows - rows_to_skip <= std::numeric_limits<size_type>::max(),
                 "The requested number of rows to read exceeds the largest cudf column size");
    return {rows_to_skip, num_source_rows - rows_to_skip};
  }
  // Limit the number of rows to the end of the input
  return {rows_to_skip,
          static_cast<size_type>(
            std::min<uint64_t>(num_rows_opt.value(), num_source_rows - rows_to_skip))};
}

}  // namespace cudf::io::detail
