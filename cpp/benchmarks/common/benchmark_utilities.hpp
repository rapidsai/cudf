/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
