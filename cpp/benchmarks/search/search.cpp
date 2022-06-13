
#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/search.hpp>
#include <cudf/types.hpp>

static constexpr cudf::size_type num_struct_members = 8;
static constexpr cudf::size_type max_str_length     = 10;

static auto create_random_structs_column(cudf::size_type n_rows)
{
  data_profile table_profile;
  table_profile.set_distribution_params(cudf::type_id::INT32, distribution_id::UNIFORM, 0, n_rows);
  table_profile.set_distribution_params(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);

  // The first two struct members are int32 and string.
  // The first column is also used as keys in groupby.
  // The subsequent struct members are int32 and string again.
  auto table = create_random_table(cycle_dtypes({cudf::type_id::INT32, cudf::type_id::STRING}, 8),
                                   row_count{n_rows},
                                   table_profile);
  return cudf::make_structs_column(n_rows, table->release(), 0, {});
}

void BM_contains(benchmark::State& state)
{
  auto const size{static_cast<cudf::size_type>(state.range(0))};

  constexpr cudf::size_type repeat_times = 10;  // <25% unique rows
  //    constexpr cudf::size_type repeat_times = 2;  // <50% unique rows
  //  constexpr int repeat_times = 1;  // <100% unique rows

  auto const needles = create_random_structs_column(size);

  auto haystack        = create_random_structs_column(size / repeat_times);
  auto const haystack0 = std::make_unique<cudf::column>(*haystack);

  for (int i = 0; i < repeat_times - 1; ++i) {
    haystack =
      cudf::concatenate(std::vector<cudf::column_view>{haystack0->view(), haystack->view()});
  }

  for ([[maybe_unused]] auto _ : state) {
    [[maybe_unused]] auto const timer = cuda_event_timer(state, true);
    auto const result                 = cudf::contains(*haystack, *needles);
  }
}

class Search : public cudf::benchmark {
};

BENCHMARK_DEFINE_F(Search, ColumnContains)(::benchmark::State& state) { BM_contains(state); }

BENCHMARK_REGISTER_F(Search, ColumnContains)
  ->RangeMultiplier(8)
  ->Ranges({{1 << 10, 1 << 26}})
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond);
