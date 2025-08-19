/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <BS_thread_pool.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <random>

template <typename key_type>
static void BM_transform_polynomials_concurrent(nvbench::state& state)
{
  auto const num_rows{static_cast<cudf::size_type>(state.get_int64("num_rows"))};
  auto const order{static_cast<cudf::size_type>(state.get_int64("order"))};
  auto const num_threads{state.get_int64("num_threads")};
  auto const runs_per_thread{state.get_int64("runs_per_thread")};

  CUDF_EXPECTS(order > 0, "Polynomial order must be greater than 0");

  data_profile profile;
  profile.set_distribution_params(cudf::type_to_id<key_type>(),
                                  distribution_id::NORMAL,
                                  static_cast<key_type>(0),
                                  static_cast<key_type>(1));
  auto column = create_random_column(cudf::type_to_id<key_type>(), row_count{num_rows}, profile);

  std::vector<std::unique_ptr<cudf::column>> constants;
  std::transform(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(order + 1),
    std::back_inserter(constants),
    [&](int) { return create_random_column(cudf::type_to_id<key_type>(), row_count{1}, profile); });

  std::vector<cudf::column_view> inputs{column->view()};
  std::transform(
    constants.begin(), constants.end(), std::back_inserter(inputs), [](auto const& col) {
      return col->view();
    });

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto root_stream = launch.get_stream().get_stream();

    auto streams = cudf::detail::fork_streams(root_stream, num_threads);
    BS::thread_pool threads(num_threads);

    nvtxRangePushA(("transform_polynomials_concurrent " + std::to_string(num_threads) +
                    " threads, " + std::to_string(runs_per_thread) + " runs/thread")
                     .c_str());

    auto polynomial_func = [&](std::size_t index) {
      // Get the appropriate stream for this thread
      auto stream_index = index % num_threads;
      auto& stream      = streams[stream_index];

      nvtxRangePushA("polynomial_transform");

      std::string type        = cudf::type_to_name(cudf::data_type{cudf::type_to_id<key_type>()});
      std::string params_decl = type + " c0";
      std::string expr        = "c0";

      for (cudf::size_type i = 1; i < order + 1; i++) {
        expr = "( " + expr + " ) * x +  c" + std::to_string(i);
        params_decl += ", " + type + " c" + std::to_string(i);
      }

      std::string udf = "__device__ inline void compute_polynomial(" + type + "* out, " + type +
                        " x, " + params_decl + ")" +
                        "{ "
                        " *out = " +
                        expr +
                        ";"
                        "}";

      cudf::transform(inputs,
                      udf,
                      cudf::data_type{cudf::type_to_id<key_type>()},
                      false,
                      std::nullopt,
                      stream.value());
      nvtxRangePop();
    };

    threads.detach_sequence(std::size_t{0},
                            static_cast<std::size_t>(static_cast<uint64_t>(num_threads) *
                                                     static_cast<uint64_t>(runs_per_thread)),
                            polynomial_func);
    threads.wait();

    // Join all the streams back to the default stream
    cudf::detail::join_streams(streams, root_stream);

    nvtxRangePop();
  });
}

#define TRANSFORM_POLYNOMIALS_CONCURRENT_BENCHMARK_DEFINE(name, key_type)                         \
  static void name(::nvbench::state& st) { ::BM_transform_polynomials_concurrent<key_type>(st); } \
  NVBENCH_BENCH(name)                                                                             \
    .set_name(#name)                                                                              \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000})                                 \
    .add_int64_axis("order", {1, 2, 4, 8})                                                        \
    .add_int64_axis("num_threads", {1, 4, 8})                                                     \
    .add_int64_axis("runs_per_thread", {10})

TRANSFORM_POLYNOMIALS_CONCURRENT_BENCHMARK_DEFINE(transform_polynomials_concurrent_float32, float);
TRANSFORM_POLYNOMIALS_CONCURRENT_BENCHMARK_DEFINE(transform_polynomials_concurrent_float64, double);
