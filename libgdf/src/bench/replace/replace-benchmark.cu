/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#include <benchmark/benchmark.h>

#include <unordered_map>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <gdf/gdf.h>

#include "../../tests/replace/utils.h"

using T = std::int64_t;

static void
BM_CPU_LoopReplace(benchmark::State &state) {
    const std::size_t length = state.range(0);

    std::vector<T> vector(length);
    thrust::sequence(vector.begin(), vector.end(), 1);

    std::vector<T> to_replace_vector(10);
    thrust::sequence(to_replace_vector.begin(), to_replace_vector.end(), 1);

    std::vector<T> values_vector(10);
    thrust::sequence(values_vector.begin(), values_vector.end(), 1);

    for (auto _ : state) {
        for (std::size_t i = 0; i < vector.size(); i++) {
            auto current = std::find(
              to_replace_vector.begin(), to_replace_vector.end(), vector[i]);
            if (current != to_replace_vector.end()) {
                std::size_t j =
                  std::distance(to_replace_vector.begin(), current);
                vector[i] = values_vector[j];
            }
        }
    }
}

static void
BM_CPU_MapReplace(benchmark::State &state) {
    const std::size_t length = state.range(0);

    std::vector<T> vector(length);
    thrust::sequence(vector.begin(), vector.end(), 1);

    std::vector<T> to_replace_vector(10);
    thrust::sequence(to_replace_vector.begin(), to_replace_vector.end(), 1);

    std::vector<T> values_vector(10);
    thrust::sequence(values_vector.begin(), values_vector.end(), 1);

    for (auto _ : state) {
        std::unordered_map<T, T> map;
        for (std::size_t i = 0; i < values_vector.size(); i++) {
            map.insert({to_replace_vector[i], values_vector[i]});
        }

        for (std::size_t i = 0; i < vector.size(); i++) {
            try {
                vector[i] = map[vector[i]];
            } catch (...) { continue; }
        }
    }
}

static void
BM_GPU_LoopReplace(benchmark::State &state) {
    const std::size_t length = state.range(0);

    thrust::device_vector<T> device_vector(length);
    thrust::sequence(device_vector.begin(), device_vector.end(), 1);
    gdf_column column = MakeGdfColumn(device_vector);

    thrust::device_vector<T> to_replace_vector(10);
    thrust::sequence(to_replace_vector.begin(), to_replace_vector.end(), 1);
    gdf_column to_replace = MakeGdfColumn(to_replace_vector);

    thrust::device_vector<T> values_vector(10);
    thrust::sequence(values_vector.begin(), values_vector.end(), 1);
    gdf_column values = MakeGdfColumn(values_vector);

    for (auto _ : state) {
        const gdf_error status =
          gdf_find_and_replace_all(&column, &to_replace, &values);
        state.PauseTiming();
        if (status != GDF_SUCCESS) { state.SkipWithError("Failed replace"); }
        state.ResumeTiming();
    }
}

BENCHMARK(BM_CPU_LoopReplace)->Ranges({{8, 8 << 16}, {8, 512}});
BENCHMARK(BM_CPU_MapReplace)->Ranges({{8, 8 << 16}, {8, 512}});
BENCHMARK(BM_GPU_LoopReplace)->Ranges({{8, 8 << 16}, {8, 512}});
