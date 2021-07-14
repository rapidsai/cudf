/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <nvbench/nvbench.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <vector>

#include "generate_input_tables.cuh"

class rmm_pool_raii {
 private:
  // memory resource factory helpers
  inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

  inline auto make_pool()
  {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
  }

  std::shared_ptr<rmm::mr::device_memory_resource> mr;

 public:
  rmm_pool_raii()
  {
    mr = make_pool();
    rmm::mr::set_current_device_resource(mr.get());  // set default resource to pool
  }

  ~rmm_pool_raii()
  {
    rmm::mr::set_current_device_resource(nullptr);
    mr.reset();
  }
};

template <typename key_type, typename payload_type, bool Nullable, typename Join>
void nvbench_join_helper(nvbench::state& state, Join JoinFunc)
{
  const cudf::size_type build_table_size{(cudf::size_type)state.get_int64("Build Table Size")};
  const cudf::size_type probe_table_size{(cudf::size_type)state.get_int64("Probe Table Size")};
  const cudf::size_type rand_max_val{build_table_size * 2};
  const double selectivity             = 0.3;
  const bool is_build_table_key_unique = true;

  // Generate build and probe tables
  cudf::test::UniformRandomGenerator<cudf::size_type> rand_gen(0, build_table_size);
  auto build_random_null_mask = [&rand_gen](int size) {
    if (Nullable) {
      // roughly 25% nulls
      auto validity = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [&rand_gen](auto i) { return (rand_gen.generate() & 3) == 0; });
      return cudf::test::detail::make_null_mask(validity, validity + size);
    } else {
      return cudf::create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    }
  };

  std::unique_ptr<cudf::column> build_key_column = [&]() {
    return Nullable ? cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<key_type>()),
                                                build_table_size,
                                                build_random_null_mask(build_table_size))
                    : cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<key_type>()),
                                                build_table_size);
  }();
  std::unique_ptr<cudf::column> probe_key_column = [&]() {
    return Nullable ? cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<key_type>()),
                                                probe_table_size,
                                                build_random_null_mask(probe_table_size))
                    : cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<key_type>()),
                                                probe_table_size);
  }();

  generate_input_tables<key_type, cudf::size_type>(
    build_key_column->mutable_view().data<key_type>(),
    build_table_size,
    probe_key_column->mutable_view().data<key_type>(),
    probe_table_size,
    selectivity,
    rand_max_val,
    is_build_table_key_unique);

  auto payload_data_it = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<payload_type> build_payload_column(
    payload_data_it, payload_data_it + build_table_size);

  cudf::test::fixed_width_column_wrapper<payload_type> probe_payload_column(
    payload_data_it, payload_data_it + probe_table_size);

  CHECK_CUDA(0);

  cudf::table_view build_table({build_key_column->view(), build_payload_column});
  cudf::table_view probe_table({probe_key_column->view(), probe_payload_column});

  // Setup join parameters and result table
  std::vector<cudf::size_type> columns_to_join = {0};

  // Benchmark the join operation
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    JoinFunc(probe_table,
             build_table,
             columns_to_join,
             columns_to_join,
             cudf::null_equality::UNEQUAL,
             stream_view);
  });
}

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_join(nvbench::state& state,
                  nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  rmm_pool_raii pool_raii;

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 std::vector<cudf::size_type> const& left_on,
                 std::vector<cudf::size_type> const& right_on,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    cudf::hash_join hj_obj(left_input.select(left_on), compare_nulls, stream);
    auto result =
      hj_obj.inner_join(right_input.select(right_on), compare_nulls, std::nullopt, stream);
    auto join_indices = std::make_pair(std::move(result.second), std::move(result.first));
  };

  nvbench_join_helper<key_type, payload_type, Nullable>(state, join);
}

// join -----------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size", {400'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size", {400'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});
