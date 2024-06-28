/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "benchmark_utilities.hpp"

void set_items_processed(::benchmark::State& state, int64_t items_processed_per_iteration)
{
  state.SetItemsProcessed(state.iterations() * items_processed_per_iteration);
}

void set_bytes_processed(::benchmark::State& state, int64_t bytes_processed_per_iteration)
{
  state.SetBytesProcessed(state.iterations() * bytes_processed_per_iteration);
}
