/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <vector>

int main(int argc, char** argv)
{
  auto input_table = make_table(5, 2);
  print_table("Original Table", input_table);

  rmm::mr::pinned_host_memory_resource phmr;

  // Pack the table into a contiguous buffer on the host
  cudf::packed_columns packed = cudf::pack(input_table, cudf::get_default_stream(), phmr);

  // Simulate copying packed data to another host
  auto copied_metadata = std::make_unique<std::vector<uint8_t>>(*packed.metadata);

  std::vector<uint8_t> copied_data(packed.gpu_data->size());
  std::memcpy(copied_data.data(), packed.gpu_data->data(), packed.gpu_data->size());

  auto device_buffer = std::make_unique<rmm::device_buffer>(
    copied_data.data(), copied_data.size(), cudf::get_default_stream(), phmr);
  cudf::packed_columns copied_packed(std::move(copied_metadata), std::move(device_buffer));

  // Unpack the packed columns back into a table view
  cudf::table_view unpacked_table = cudf::unpack(copied_packed);
  print_table("Host Copied Unpacked Table", unpacked_table);

  return 0;
}
