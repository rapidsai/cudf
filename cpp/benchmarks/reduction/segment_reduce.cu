#include "nvbench/enum_type_list.cuh"
#include "rmm/cuda_stream_view.hpp"
#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <fixture/benchmark_fixture.hpp>
#include <nvbench/nvbench.cuh>
#include <synchronization/synchronization.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/device_vector.h>

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudf {

bool constexpr is_boolean_output_agg(aggregation::Kind kind)
{
  return kind == aggregation::ALL || kind == aggregation::ANY;
}

template <aggregation::Kind kind>
std::unique_ptr<aggregation> make_simple_aggregation()
{
  switch (kind) {
    case aggregation::SUM: return make_sum_aggregation();
    case aggregation::PRODUCT: return make_product_aggregation();
    case aggregation::MIN: return make_min_aggregation();
    case aggregation::MAX: return make_max_aggregation();
    case aggregation::ALL: return make_all_aggregation();
    case aggregation::ANY: return make_any_aggregation();
    default: CUDF_FAIL("Unsupported simple segmented aggregation");
  }
}

template <typename InputType>
std::pair<std::unique_ptr<column>, thrust::device_vector<size_type>> make_test_data(
  nvbench::state& state)
{
  auto const column_size{size_type(state.get_int64("column_size"))};
  auto const num_segments{size_type(state.get_int64("num_segments"))};

  auto segment_length = column_size / num_segments;

  test::UniformRandomGenerator<InputType> rand_gen(0, 100);
  auto data_it = detail::make_counting_transform_iterator(
    0, [&rand_gen](auto i) { return rand_gen.generate(); });

  auto offset_it =
    detail::make_counting_transform_iterator(0, [&column_size, &segment_length](auto i) {
      return column_size < i * segment_length ? column_size : i * segment_length;
    });

  test::fixed_width_column_wrapper<InputType> input(data_it, data_it + column_size);
  std::vector<size_type> h_offsets(offset_it, offset_it + num_segments + 1);
  thrust::device_vector<size_type> d_offsets(h_offsets);

  return std::make_pair(input.release(), d_offsets);
}

template <typename InputType, typename OutputType, aggregation::Kind kind>
std::enable_if_t<!is_boolean_output_agg(kind) || std::is_same_v<OutputType, bool>, void>
BM_Simple_Segmented_Reduction(nvbench::state& state,
                              nvbench::type_list<InputType, OutputType, nvbench::enum_type<kind>>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii rmm_pool;

  auto const column_size{size_type(state.get_int64("column_size"))};
  auto [input, offsets] = make_test_data<InputType>(state);
  // auto agg              = make_simple_aggregation<kind>();
  auto agg = make_sum_aggregation();

  state.add_element_count(column_size);
  state.add_global_memory_reads<InputType>(column_size);
  state.add_global_memory_writes<OutputType>(column_size);

  state.exec(
    nvbench::exec_tag::sync,
    [input_view = input->view(), offset_span = device_span<size_type>{offsets}, &agg](
      nvbench::launch& launch) {
      segmented_reduce(
        input_view, offset_span, *agg, data_type{type_to_id<OutputType>()}, null_policy::INCLUDE);
    });
}

template <typename InputType, typename OutputType, aggregation::Kind kind>
std::enable_if_t<is_boolean_output_agg(kind) && !std::is_same_v<OutputType, bool>, void>
BM_Simple_Segmented_Reduction(nvbench::state& state,
                              nvbench::type_list<InputType, OutputType, nvbench::enum_type<kind>>)
{
  state.skip("Invalid combination of dtype and aggregation type.");
}

using Types = nvbench::type_list<bool, int32_t, float, double>;
// Skip benchmarking MAX/ANY since they are covered by MIN/ALL respectively.
using AggKinds = nvbench::
  enum_type_list<aggregation::SUM, aggregation::PRODUCT, aggregation::MIN, aggregation::ALL>;

NVBENCH_BENCH_TYPES(BM_Simple_Segmented_Reduction, NVBENCH_TYPE_AXES(Types, Types, AggKinds))
  .set_name("segmented_reduction_simple")
  .set_type_axes_names({"InputType", "OutputType", "AggregationKinds"})
  .add_int64_axis("column_size", {100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_segments", {1'000, 10'000, 100'000});

}  // namespace cudf
