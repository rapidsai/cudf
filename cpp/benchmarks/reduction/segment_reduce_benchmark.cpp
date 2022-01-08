#include "cudf/column/column.hpp"
#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <fixture/benchmark_fixture.hpp>
#include <memory>
#include <synchronization/synchronization.hpp>
#include <unordered_map>
#include <utility>

namespace cudf {

class SegmentReduction : public cudf::benchmark {
};

template <typename InputType>
void BM_Segment_Reduction_Simple(::benchmark::State& state, std::unique_ptr<aggregation> const& agg)
{
  auto const column_size{size_type(state.range(0))};
  auto const num_segments{size_type(state.range(1))};
  auto const segment_length = column_size / num_segments;

  test::UniformRandomGenerator<InputType> rand_gen(0, 100);
  auto data_it = detail::make_counting_transform_iterator(
    0, [&rand_gen](auto i) { return rand_gen.generate(); });

  auto offset_it = detail::make_counting_transform_iterator(
    0,
    [&column_size, &segment_length](auto i) { return std::min(column_size, i * segment_length); });

  test::fixed_width_column_wrapper<InputType> input(data_it, data_it + column_size);
  test::fixed_width_column_wrapper<size_type> offsets(offset_it, offset_it + num_segments + 1);

  for (auto _ : state) {
    cuda_event_timer timer(state, true);
    auto result = cudf::segmented_reduce(input,
                                         offsets,
                                         make_sum_aggregation(),
                                         data_type{type_to_id<InputType>()},
                                         null_policy::EXCLUDE);
  }
}

#define concat(a, b, c) a##b##c
#define get_agg(op)     concat(cudf::make_, op, _aggregation())

#define SRBM_DEFINE(name, type, op)                                                            \
  BENCHMARK_DEFINE_F(SegmentReduction, name)(::benchmark::State & state)                       \
  {                                                                                            \
    BM_Segment_Reduction_Simple<type>(state, get_agg(op));                                     \
  }                                                                                            \
  BENCHMARK_REGISTER_F(SegmentReduction, name)                                                 \
    ->UseManualTime()                                                                          \
    ->ArgsProduct(                                                                             \
      {{1 << 14, /*10k*/ 1 << 17, /*100k*/ 1 << 20, /*1M*/ 1 << 24, /*10M*/ 1 << 27 /*100M*/}, \
       {                                                                                       \
         128,                                                                                  \
         1024,                                                                                 \
       }});

#define REDUCE_BENCHMARK_DEFINE(type, aggregation) \
  SRBM_DEFINE(concat(type, _, aggregation), type, aggregation)

#define REDUCE_BENCHMARK_NUMERIC(aggregation)    \
  REDUCE_BENCHMARK_DEFINE(bool, aggregation);    \
  REDUCE_BENCHMARK_DEFINE(int8_t, aggregation);  \
  REDUCE_BENCHMARK_DEFINE(int16_t, aggregation); \
  REDUCE_BENCHMARK_DEFINE(int32_t, aggregation); \
  REDUCE_BENCHMARK_DEFINE(int64_t, aggregation); \
  REDUCE_BENCHMARK_DEFINE(float, aggregation);   \
  REDUCE_BENCHMARK_DEFINE(double, aggregation);

REDUCE_BENCHMARK_NUMERIC(sum)
REDUCE_BENCHMARK_NUMERIC(product)
REDUCE_BENCHMARK_NUMERIC(min)
REDUCE_BENCHMARK_NUMERIC(max)
REDUCE_BENCHMARK_NUMERIC(any)
REDUCE_BENCHMARK_NUMERIC(all)

}  // namespace cudf
