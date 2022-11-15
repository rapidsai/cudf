/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/json.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <thrust/random.h>

class JsonPath : public cudf::benchmark {
};

const std::vector<std::string> Books{
  R"json({
"category": "reference",
"author": "Nigel Rees",
"title": "Sayings of the Century",
"price": 8.95
})json",
  R"json({
"category": "fiction",
"author": "Evelyn Waugh",
"title": "Sword of Honour",
"price": 12.99
})json",
  R"json({
"category": "fiction",
"author": "Herman Melville",
"title": "Moby Dick",
"isbn": "0-553-21311-3",
"price": 8.99
})json",
  R"json({
"category": "fiction",
"author": "J. R. R. Tolkien",
"title": "The Lord of the Rings",
"isbn": "0-395-19395-8",
"price": 22.99
})json"};
constexpr int Approx_book_size = 110;
const std::vector<std::string> Bicycles{
  R"json({"color": "red", "price": 9.95})json",
  R"json({"color": "green", "price": 29.95})json",
  R"json({"color": "blue", "price": 399.95})json",
  R"json({"color": "yellow", "price": 99.95})json",
  R"json({"color": "mauve", "price": 199.95})json",
};
constexpr int Approx_bicycle_size = 33;
std::string Misc{"\n\"expensive\": 10\n"};

struct json_benchmark_row_builder {
  int const desired_bytes;
  cudf::size_type const num_rows;
  cudf::column_device_view const d_books_bicycles[2];  // Books, Bicycles strings
  cudf::column_device_view const d_book_pct;           // Book percentage
  cudf::column_device_view const d_misc_order;         // Misc-Store order
  cudf::column_device_view const d_store_order;        // Books-Bicycles order
  int32_t* d_offsets{};
  char* d_chars{};
  thrust::minstd_rand rng{5236};
  thrust::uniform_int_distribution<int> dist{};

  // internal data structure for {bytes, out_ptr} with operator+=
  struct bytes_and_ptr {
    cudf::size_type bytes;
    char* ptr;
    __device__ bytes_and_ptr& operator+=(cudf::string_view const& str_append)
    {
      bytes += str_append.size_bytes();
      if (ptr) { ptr = cudf::strings::detail::copy_string(ptr, str_append); }
      return *this;
    }
  };

  __device__ inline void copy_items(int this_idx,
                                    cudf::size_type num_items,
                                    bytes_and_ptr& output_str)
  {
    using param_type = thrust::uniform_int_distribution<int>::param_type;
    dist.param(param_type{0, d_books_bicycles[this_idx].size() - 1});
    cudf::string_view comma(",\n", 2);
    for (int i = 0; i < num_items; i++) {
      if (i > 0) { output_str += comma; }
      int idx   = dist(rng);
      auto item = d_books_bicycles[this_idx].element<cudf::string_view>(idx);
      output_str += item;
    }
  }

  __device__ void operator()(cudf::size_type idx)
  {
    int num_books       = 2;
    int num_bicycles    = 2;
    int remaining_bytes = max(
      0, desired_bytes - ((num_books * Approx_book_size) + (num_bicycles * Approx_bicycle_size)));

    // divide up the remainder between books and bikes
    auto book_pct = d_book_pct.element<float>(idx);
    // {Misc, store} OR {store, Misc}
    // store: {books, bicycles} OR store: {bicycles, books}
    float bicycle_pct = 1.0f - book_pct;
    num_books += (remaining_bytes * book_pct) / Approx_book_size;
    num_bicycles += (remaining_bytes * bicycle_pct) / Approx_bicycle_size;

    char* out_ptr = d_chars ? d_chars + d_offsets[idx] : nullptr;
    bytes_and_ptr output_str{0, out_ptr};
    //
    cudf::string_view comma(",\n", 2);
    cudf::string_view brace1("{\n", 2);
    cudf::string_view store_member_start[2]{{"\"book\": [\n", 10}, {"\"bicycle\": [\n", 13}};
    cudf::string_view store("\"store\": {\n", 11);
    cudf::string_view Misc{"\"expensive\": 10", 15};
    cudf::string_view brace2("\n}", 2);
    cudf::string_view square2{"\n]", 2};

    output_str += brace1;
    if (d_misc_order.element<bool>(idx)) {  // Misc. first.
      output_str += Misc;
      output_str += comma;
    }
    output_str += store;
    for (int store_order = 0; store_order < 2; store_order++) {
      if (store_order > 0) { output_str += comma; }
      int this_idx    = (d_store_order.element<bool>(idx) == store_order);
      auto& mem_start = store_member_start[this_idx];
      output_str += mem_start;
      copy_items(this_idx, this_idx == 0 ? num_books : num_bicycles, output_str);
      output_str += square2;
    }
    output_str += brace2;
    if (!d_misc_order.element<bool>(idx)) {  // Misc, if not first.
      output_str += comma;
      output_str += Misc;
    }
    output_str += brace2;
    if (!output_str.ptr) d_offsets[idx] = output_str.bytes;
  }
};

auto build_json_string_column(int desired_bytes, int num_rows)
{
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::FLOAT32, distribution_id::UNIFORM, 0.0, 1.0);
  auto float_2bool_columns =
    create_random_table({cudf::type_id::FLOAT32, cudf::type_id::BOOL8, cudf::type_id::BOOL8},
                        row_count{num_rows},
                        profile);

  cudf::test::strings_column_wrapper books(Books.begin(), Books.end());
  cudf::test::strings_column_wrapper bicycles(Bicycles.begin(), Bicycles.end());
  auto d_books       = cudf::column_device_view::create(books);
  auto d_bicycles    = cudf::column_device_view::create(bicycles);
  auto d_book_pct    = cudf::column_device_view::create(float_2bool_columns->get_column(0));
  auto d_misc_order  = cudf::column_device_view::create(float_2bool_columns->get_column(1));
  auto d_store_order = cudf::column_device_view::create(float_2bool_columns->get_column(2));
  json_benchmark_row_builder jb{
    desired_bytes, num_rows, {*d_books, *d_bicycles}, *d_book_pct, *d_misc_order, *d_store_order};
  auto children = cudf::strings::detail::make_strings_children(
    jb, num_rows, cudf::get_default_stream(), rmm::mr::get_current_device_resource());
  return cudf::make_strings_column(
    num_rows, std::move(children.first), std::move(children.second), 0, {});
}

void BM_case(benchmark::State& state, std::string query_arg)
{
  srand(5236);
  int num_rows      = state.range(0);
  int desired_bytes = state.range(1);
  auto input        = build_json_string_column(desired_bytes, num_rows);
  cudf::strings_column_view scv(input->view());
  size_t num_chars = scv.chars().size();

  std::string json_path(query_arg);

  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto result = cudf::strings::get_json_object(scv, json_path);
    cudaStreamSynchronize(0);
  }

  // this isn't strictly 100% accurate. a given query isn't necessarily
  // going to visit every single incoming character.  but in spirit it does.
  state.SetBytesProcessed(state.iterations() * num_chars);
}

#define JSON_BENCHMARK_DEFINE(name, query)                                                  \
  BENCHMARK_DEFINE_F(JsonPath, name)(::benchmark::State & state) { BM_case(state, query); } \
  BENCHMARK_REGISTER_F(JsonPath, name)                                                      \
    ->ArgsProduct({{100, 1000, 100000, 400000}, {300, 600, 4096}})                          \
    ->UseManualTime()                                                                       \
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
