/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

int main(int argc, char** argv)
{
  auto input_table = make_table(5, 2);
  print_table("Original Table", input_table);

  rmm::mr::pinned_host_memory_resource phmr;

  // Pack the table into a contiguous buffer on the host
  cudf::packed_columns packed = cudf::pack(input_table, cudf::get_default_stream(), phmr);

  // Unpack the packed columns back into a table view
  cudf::table_view unpacked_table = cudf::unpack(packed);
  print_table("Host Unpacked Table", unpacked_table);

  return 0;
}
