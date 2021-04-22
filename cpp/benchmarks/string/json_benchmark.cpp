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

#include <benchmark/benchmark.h>
#include <benchmarks/common/generate_benchmark_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/json.hpp>
#include <cudf/strings/strings_column_view.hpp>

class JsonPath : public cudf::benchmark {
};

float frand() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }

int rand_range(int min, int max) { return min + static_cast<int>(frand() * (max - min)); }

std::vector<std::string> Books{
  "{\n\"category\": \"reference\",\n\"author\": \"Nigel Rees\",\n\"title\": \"Sayings of the "
  "Century\",\n\"price\": 8.95\n}",
  "{\n\"category\": \"fiction\",\n\"author\": \"Evelyn Waugh\",\n\"title\": \"Sword of "
  "Honour\",\n\"price\": 12.99\n}",
  "{\n\"category\": \"fiction\",\n\"author\": \"Herman Melville\",\n\"title\": \"Moby "
  "Dick\",\n\"isbn\": \"0-553-21311-3\",\n\"price\": 8.99\n}",
  "{\n\"category\": \"fiction\",\n\"author\": \"J. R. R. Tolkien\",\n\"title\": \"The Lord of the "
  "Rings\",\n\"isbn\": \"0-395-19395-8\",\n\"price\": 22.99\n}"};
constexpr int Approx_book_size = 110;
std::vector<std::string> Bicycles{
  "{\"color\": \"red\", \"price\": 9.95}",
  "{\"color\": \"green\", \"price\": 29.95}",
  "{\"color\": \"blue\", \"price\": 399.95}",
  "{\"color\": \"yellow\", \"price\": 99.95}",
  "{\"color\": \"mauve\", \"price\": 199.95}",
};
constexpr int Approx_bicycle_size = 33;
std::string Misc{"\n\"expensive\": 10\n"};
std::string generate_field(std::vector<std::string> const& values, int num_values)
{
  std::string res;
  for (int idx = 0; idx < num_values; idx++) {
    if (idx > 0) { res += std::string(",\n"); }
    int vindex = std::min(static_cast<int>(floor(frand() * values.size())),
                          static_cast<int>(values.size() - 1));
    res += values[vindex];
  }
  return res;
}

std::string build_row(int desired_bytes)
{
  // always have at least 2 books and 2 bikes
  int num_books    = 2;
  int num_bicycles = 2;
  int remaining_bytes =
    desired_bytes - ((num_books * Approx_book_size) + (num_bicycles * Approx_bicycle_size));

  // divide up the remainder between books and bikes
  float book_pct    = frand();
  float bicycle_pct = 1.0f - book_pct;
  num_books += (remaining_bytes * book_pct) / Approx_book_size;
  num_bicycles += (remaining_bytes * bicycle_pct) / Approx_bicycle_size;

  std::string books    = "\"book\": [\n" + generate_field(Books, num_books) + "]\n";
  std::string bicycles = "\"bicycle\": [\n" + generate_field(Bicycles, num_bicycles) + "]\n";

  std::string store = "\"store\": {\n";
  if (frand() <= 0.5f) {
    store += books + std::string(",\n") + bicycles;
  } else {
    store += bicycles + std::string(",\n") + books;
  }
  store += std::string("}\n");

  std::string row = std::string("{\n");
  if (frand() <= 0.5f) {
    row += store + std::string(",\n") + Misc;
  } else {
    row += Misc + std::string(",\n") + store;
  }
  row += std::string("}\n");
  return row;
}

template <class... QueryArg>
static void BM_case(benchmark::State& state, QueryArg&&... query_arg)
{
  srand(5236);
  auto iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [desired_bytes = state.range(1)](int index) { return build_row(desired_bytes); });
  int num_rows = state.range(0);
  cudf::test::strings_column_wrapper input(iter, iter + num_rows);
  cudf::strings_column_view scv(input);
  size_t num_chars = scv.chars().size();

  std::string json_path(query_arg...);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto result = cudf::strings::get_json_object(scv, json_path);
    cudaStreamSynchronize(0);
  }

  // this isn't strictly 100% accurate. a given query isn't necessarily
  // going to visit every single incoming character.  but in spirit it does.
  state.SetBytesProcessed(state.iterations() * num_chars);
}

#define JSON_BENCHMARK_DEFINE(name, query)                         \
  BENCHMARK_CAPTURE(BM_case, name, query)                          \
    ->ArgsProduct({{100, 1000, 100000, 400000}, {300, 600, 4096}}) \
    ->UseManualTime()                                              \
    ->Unit(benchmark::kMillisecond);

JSON_BENCHMARK_DEFINE(query0, "$");
JSON_BENCHMARK_DEFINE(query1, "$.store");
JSON_BENCHMARK_DEFINE(query2, "$.store.book");
JSON_BENCHMARK_DEFINE(query3, "$.store.*");
JSON_BENCHMARK_DEFINE(query4, "$.store.book[*]");
JSON_BENCHMARK_DEFINE(query5, "$.store.book[*].category");
JSON_BENCHMARK_DEFINE(query6, "$.store['bicycle']");
JSON_BENCHMARK_DEFINE(query7, "$.store.book[*]['isbn']");
JSON_BENCHMARK_DEFINE(query8, "$.store.bicycle[1]");
