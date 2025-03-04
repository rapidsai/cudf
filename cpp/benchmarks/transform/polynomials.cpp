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
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <random>

template <typename key_type>
static void BM_transform_polynomials(nvbench::state& state)
{
  auto const num_rows{static_cast<cudf::size_type>(state.get_int64("num_rows"))};
  auto const order{static_cast<cudf::size_type>(state.get_int64("order"))};

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

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(num_rows);
  state.add_global_memory_writes<key_type>(num_rows);

  std::vector<cudf::column_view> inputs{*column};
  std::transform(constants.begin(),
                 constants.end(),
                 std::back_inserter(inputs),
                 [](auto& col) -> cudf::column_view { return *col; });

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // computes polynomials: (((ax + b)x + c)x + d)x + e... = ax**4 + bx**3 + cx**2 + dx + e....

    cudf::scoped_range range{"benchmark_iteration"};

    std::string type = cudf::type_to_name(cudf::data_type{cudf::type_to_id<key_type>()});

    std::string params_decl = type + " c0";
    std::string expr        = "c0";

    for (cudf::size_type i = 1; i < order + 1; i++) {
      expr = "( " + expr + " ) * x +  c" + std::to_string(i);
      params_decl += ", " + type + " c" + std::to_string(i);
    }

    static_assert(std::is_same_v<key_type, float> || std::is_same_v<key_type, double>);

    // clang-format off
    std::string udf =
    "__device__ inline void compute_polynomial(" + type + "* out, " + type + " x, " + params_decl + ")" +
"{ "
" *out = " + expr + ";"
"}";

    // clang-format on

    cudf::transform(inputs,
                    udf,
                    cudf::data_type{cudf::type_to_id<key_type>()},
                    false,
                    launch.get_stream().get_stream());
  });
}

#define TRANSFORM_POLYNOMIALS_BENCHMARK_DEFINE(name, key_type)                         \
                                                                                       \
  static void name(::nvbench::state& st) { ::BM_transform_polynomials<key_type>(st); } \
  NVBENCH_BENCH(name)                                                                  \
    .set_name(#name)                                                                   \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})         \
    .add_int64_axis("order", {1, 2, 4, 8, 16, 32})

TRANSFORM_POLYNOMIALS_BENCHMARK_DEFINE(transform_polynomials_float32, float);

TRANSFORM_POLYNOMIALS_BENCHMARK_DEFINE(transform_polynomials_float64, double);
