/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_utilities.hpp"

void set_items_processed(::benchmark::State& state, int64_t items_processed_per_iteration)
{
  state.SetItemsProcessed(state.iterations() * items_processed_per_iteration);
}

void set_bytes_processed(::benchmark::State& state, int64_t bytes_processed_per_iteration)
{
  state.SetBytesProcessed(state.iterations() * bytes_processed_per_iteration);
}
