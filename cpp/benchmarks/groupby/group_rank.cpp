/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

template <cudf::rank_method method>
static void nvbench_groupby_rank(nvbench::state& state,
                                 nvbench::type_list<nvbench::enum_type<method>>)
{
  constexpr auto dtype = cudf::type_to_id<int64_t>();

  bool const is_sorted              = state.get_int64("is_sorted");
  cudf::size_type const column_size = state.get_int64("data_size");
  constexpr int num_groups          = 100;

  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    dtype, distribution_id::UNIFORM, 0, num_groups);

  auto source_table = create_random_table({dtype, dtype}, row_count{column_size}, profile);

  // values to be pre-sorted too for groupby rank
  if (is_sorted) source_table = cudf::sort(*source_table);

  cudf::table_view keys{{source_table->view().column(0)}};
  cudf::column_view order_by{source_table->view().column(1)};

  auto agg = cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(method);
  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back(cudf::groupby::scan_request());
  requests[0].values = order_by;
  requests[0].aggregations.push_back(std::move(agg));

  cudf::groupby::groupby gb_obj(
    keys, cudf::null_policy::EXCLUDE, is_sorted ? cudf::sorted::YES : cudf::sorted::NO);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    // groupby scan uses sort implementation
    auto result = gb_obj.scan(requests);
  });
}

enum class rank_method : int32_t {};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::rank_method,
  [](cudf::rank_method value) {
    switch (value) {
      case cudf::rank_method::FIRST: return "FIRST";
      case cudf::rank_method::AVERAGE: return "AVERAGE";
      case cudf::rank_method::MIN: return "MIN";
      case cudf::rank_method::MAX: return "MAX";
      case cudf::rank_method::DENSE: return "DENSE";
      default: return "unknown";
    }
  },
  [](cudf::rank_method value) {
    switch (value) {
      case cudf::rank_method::FIRST: return "cudf::rank_method::FIRST";
      case cudf::rank_method::AVERAGE: return "cudf::rank_method::AVERAGE";
      case cudf::rank_method::MIN: return "cudf::rank_method::MIN";
      case cudf::rank_method::MAX: return "cudf::rank_method::MAX";
      case cudf::rank_method::DENSE: return "cudf::rank_method::DENSE";
      default: return "unknown";
    }
  })

using methods = nvbench::enum_type_list<cudf::rank_method::AVERAGE,
                                        cudf::rank_method::DENSE,
                                        cudf::rank_method::FIRST,
                                        cudf::rank_method::MAX,
                                        cudf::rank_method::MIN>;

NVBENCH_BENCH_TYPES(nvbench_groupby_rank, NVBENCH_TYPE_AXES(methods))
  .set_type_axes_names({"rank_method"})
  .set_name("groupby_rank")
  .add_int64_axis("data_size",
                  {
                    1000000,    // 1M
                    10000000,   // 10M
                    100000000,  // 100M
                  })

  .add_int64_axis("is_sorted", {0, 1});
