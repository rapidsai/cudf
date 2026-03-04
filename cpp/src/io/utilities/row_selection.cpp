/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/row_selection.hpp"

#include <algorithm>

namespace cudf::io::detail {

std::pair<int64_t, int64_t> skip_rows_num_rows_from_options(int64_t skip_rows,
                                                            std::optional<int64_t> const& num_rows,
                                                            int64_t num_source_rows)
{
  auto const rows_to_skip      = std::min(skip_rows, num_source_rows);
  auto const num_rows_can_read = num_source_rows - rows_to_skip;

  if (not num_rows.has_value()) { return {rows_to_skip, num_rows_can_read}; }

  // Limit the number of rows to the end of the input
  return {rows_to_skip, std::min(num_rows.value(), num_rows_can_read)};
}

}  // namespace cudf::io::detail
