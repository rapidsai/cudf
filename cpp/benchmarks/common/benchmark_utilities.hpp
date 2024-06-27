/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>

/**
 * @brief Sets the number of items processed during the benchmark.
 * 
 * This function could be used instead of ::benchmark::State.SetItemsProcessed()
 * to avoid repeatedly computing ::benchmark::State.iterations() * items_processed_per_iteration.
 * 
 * @param state the benchmark state
 * @param items_processed_per_iteration number of items processed per iteration
 */
void set_items_processed(::benchmark::State& state, int64_t items_processed_per_iteration);

/**
 * @brief Sets the number of bytes processed during the benchmark.
 * 
 * This function could be used instead of ::benchmark::State.SetItemsProcessed()
 * to avoid repeatedly computing ::benchmark::State.iterations() * bytes_processed_per_iteration.
 * 
 * @param state the benchmark state
 * @param bytes_processed_per_iteration number of bytes processed per iteration
 */
void set_bytes_processed(::benchmark::State& state, int64_t bytes_processed_per_iteration);
