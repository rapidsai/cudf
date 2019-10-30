/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/column_wrapper_factory.hpp>

#include <cudf/reshape.hpp>
#include <cudf/types.h>

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>
#include <random>

template <class T>
class Reshape : public ::benchmark::Fixture {
public:
  using TypeParam = T;
};

// TODO: put it in a struct so `uniform` can be remade with different min, max
template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

template <class T>
void BM_stack(benchmark::State& state){
  using wrapper = cudf::test::column_wrapper<T>;

  const gdf_size_type num_columns{(gdf_size_type)state.range(0)};
  const gdf_size_type column_size{(gdf_size_type)state.range(1)};

  std::vector<wrapper> columns;

  auto make_table = [&] (std::vector<wrapper>& cols,
                         gdf_size_type col_size) -> cudf::table
  {
    cudf::test::column_wrapper_factory<T> factory;

    for (gdf_size_type i = 0; i < num_columns; i++) {
      cols.emplace_back(factory.make(col_size,
        [=](cudf::size_type row) { return random_int(0, 10); }
        ,[=](cudf::size_type row) { return random_int(0, 10) < 90; }
      ));
    }

    std::vector<gdf_column*> raw_cols(num_columns, nullptr);
    std::transform(cols.begin(), cols.end(), raw_cols.begin(),
                   [](wrapper &c) { return c.get(); });

    return cudf::table{raw_cols.data(), num_columns};
  };

  auto table = make_table(columns, column_size);

  for(auto _ : state){
    cuda_event_timer timer(state, true);
    auto col = cudf::stack(table);
    gdf_column_free(&col);
  }
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int num_cols = 2; num_cols <= 10; num_cols*=2)
    for (int col_size = 1000; col_size <= 1000000; col_size *= 10)
      b->Args({num_cols, col_size});
}

BENCHMARK_TEMPLATE_DEFINE_F(Reshape, IntStack, int64_t)
(::benchmark::State& state) {
  BM_stack<TypeParam>(state);
}

BENCHMARK_REGISTER_F(Reshape, IntStack)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Apply(CustomArguments);


BENCHMARK_TEMPLATE_DEFINE_F(Reshape, StrStack, cudf::nvstring_category)
(::benchmark::State& state) {
  BM_stack<TypeParam>(state);
}

BENCHMARK_REGISTER_F(Reshape, StrStack)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Apply(CustomArguments);


/**
 * @brief Benchmarks for approach towards combining NVCategories
 * 
 * The below benchmarks are to compare the performance of two approaches towards
 * combining NVCategories. One is a cumulative call to `NVCategory.merge_and_remap`
 * for each pair of columns. The other is to call `NVCategory::create_from_categories`
 * on all the categories together.
 * 
 * To run the benchmarks, uncomment the following #define
 */

// #define COMPARE_CATEGORY_MERGE

class String : public ::benchmark::Fixture {};

void BM_mar(benchmark::State& state){
  using wrapper = cudf::test::column_wrapper<cudf::nvstring_category>;

  const gdf_size_type num_columns{(gdf_size_type)state.range(0)};
  const gdf_size_type column_size{(gdf_size_type)state.range(1)};

  std::vector<wrapper> columns;

  cudf::test::column_wrapper_factory<cudf::nvstring_category> factory;

  for (gdf_size_type i = 0; i < num_columns; i++) {
    columns.emplace_back(factory.make(column_size,
      [=](cudf::size_type row) { return random_int(0, 10); }
      ,[=](cudf::size_type row) { return random_int(0, 10) < 90; }
    ));
  }

  std::vector<gdf_column*> raw_cols(num_columns, nullptr);
  std::transform(columns.begin(), columns.end(), raw_cols.begin(),
                  [](wrapper &c) { return c.get(); });

  for(auto _ : state){
    cuda_event_timer timer(state, true);
    NVCategory * combined_category = static_cast<NVCategory *>(raw_cols[0]->dtype_info.category);

    for(int column_index = 1; column_index < num_columns; column_index++){
      NVCategory * temp = combined_category;
      if(raw_cols[column_index]->size > 0){
        gdf_column* in_col = const_cast<gdf_column*>(raw_cols[column_index]);
        combined_category = combined_category->merge_and_remap(
            * static_cast<NVCategory *>(
                in_col->dtype_info.category));
        if(column_index > 1){
          NVCategory::destroy(temp);
        }
      }
    }

    NVCategory::destroy(combined_category);
  }
}

BENCHMARK_DEFINE_F(String, MAR)(::benchmark::State& state) {
  BM_mar(state);
}

#ifdef COMPARE_CATEGORY_MERGE
BENCHMARK_REGISTER_F(String, MAR)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Apply(CustomArguments);
#endif

void BM_cfc(benchmark::State& state){
  using wrapper = cudf::test::column_wrapper<cudf::nvstring_category>;

  const gdf_size_type num_columns{(gdf_size_type)state.range(0)};
  const gdf_size_type column_size{(gdf_size_type)state.range(1)};

  std::vector<wrapper> columns;

  cudf::test::column_wrapper_factory<cudf::nvstring_category> factory;

  for (gdf_size_type i = 0; i < num_columns; i++) {
    columns.emplace_back(factory.make(column_size,
      [=](cudf::size_type row) { return random_int(0, 10); }
      ,[=](cudf::size_type row) { return random_int(0, 10) < 90; }
    ));
  }

  std::vector<gdf_column*> raw_cols(num_columns, nullptr);
  std::transform(columns.begin(), columns.end(), raw_cols.begin(),
                  [](wrapper &c) { return c.get(); });

  for(auto _ : state){
    cuda_event_timer timer(state, true);
    std::vector<NVCategory *> cats;
    std::transform(raw_cols.begin(), raw_cols.end(), std::back_inserter(cats), 
      [&](gdf_column* c) { return static_cast<NVCategory *>(c->dtype_info.category); });

    NVCategory * combined_category = NVCategory::create_from_categories(cats);
    NVCategory::destroy(combined_category);
  }
}

BENCHMARK_DEFINE_F(String, CFC)(::benchmark::State& state) {
  BM_cfc(state);
}

#ifdef COMPARE_CATEGORY_MERGE
BENCHMARK_REGISTER_F(String, CFC)
  ->UseManualTime()
  ->Unit(benchmark::kMillisecond)
  ->Apply(CustomArguments);
#endif
