/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

int main(int argc, char** argv)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(50)};
  cudf::set_current_device_resource(&mr);

  auto input_table = make_table(5, 2);
  print_table("Original Table", input_table);

  // Pack the table into a contiguous buffer on the device
  cudf::packed_columns packed = cudf::pack(input_table);

  // Unpack the packed columns back into a table view
  cudf::table_view unpacked_table = cudf::unpack(packed);
  print_table("Device Unpacked Table", unpacked_table);

  return 0;
}
