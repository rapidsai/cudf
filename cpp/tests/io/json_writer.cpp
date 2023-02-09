/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <string>
#include <vector>

struct JsonWriterTest : public cudf::test::BaseFixture {
};

TEST_F(JsonWriterTest, EmptyInput)
{
  cudf::test::strings_column_wrapper col1;
  cudf::test::strings_column_wrapper col2;
  cudf::test::fixed_width_column_wrapper<int> col3;
  cudf::test::fixed_width_column_wrapper<float> col4;
  cudf::test::fixed_width_column_wrapper<int16_t> col5;
  cudf::table_view tbl_view{{col1, col2, col3, col4}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int"}, {"float"}, {"int16"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(false)
                       .na_rep("null")
                       .build();

  // Empty columns in table
  cudf::io::write_json(out_options, rmm::mr::get_current_device_resource());
  std::string const expected = R"([])";
  EXPECT_EQ(expected, std::string(out_buffer.data(), out_buffer.size()));

  // Empty columns in table - JSON Lines
  out_buffer.clear();
  out_options.enable_lines(true);
  cudf::io::write_json(out_options, rmm::mr::get_current_device_resource());
  std::string const expected_lines = "\n";
  EXPECT_EQ(expected_lines, std::string(out_buffer.data(), out_buffer.size()));

  // Empty table - JSON Lines
  cudf::table_view tbl_view2{};
  out_options.set_table(tbl_view2);
  out_buffer.clear();
  cudf::io::write_json(out_options, rmm::mr::get_current_device_resource());
  EXPECT_EQ(expected_lines, std::string(out_buffer.data(), out_buffer.size()));
}

TEST_F(JsonWriterTest, ErrorCases)
{
  cudf::test::strings_column_wrapper col1{"a", "b", "c"};
  cudf::test::strings_column_wrapper col2{"d", "e", "f"};
  cudf::test::fixed_width_column_wrapper<int> col3{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> col4{1.5, 2.5, 3.5};
  cudf::test::fixed_width_column_wrapper<int16_t> col5{{1, 2, 3},
                                                       cudf::test::iterators::nulls_at({0, 2})};
  cudf::table_view tbl_view{{col1, col2, col3, col4, col5}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int"}, {"float"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(false)
                       .na_rep("null")
                       .build();

  // not enough column names
  EXPECT_THROW(cudf::io::write_json(out_options, rmm::mr::get_current_device_resource()),
               cudf::logic_error);

  mt.schema_info.emplace_back("int16");
  out_options.set_metadata(mt);
  EXPECT_NO_THROW(cudf::io::write_json(out_options, rmm::mr::get_current_device_resource()));

  // chunk_rows must be at least 8
  out_options.set_rows_per_chunk(0);
  EXPECT_THROW(cudf::io::write_json(out_options, rmm::mr::get_current_device_resource()),
               cudf::logic_error);
}

TEST_F(JsonWriterTest, PlainTable)
{
  cudf::test::strings_column_wrapper col1{"a", "b", "c"};
  cudf::test::strings_column_wrapper col2{"d", "e", "f"};
  cudf::test::fixed_width_column_wrapper<int> col3{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> col4{1.5, 2.5, 3.5};
  cudf::test::fixed_width_column_wrapper<int16_t> col5{{1, 2, 3},
                                                       cudf::test::iterators::nulls_at({0, 2})};
  cudf::table_view tbl_view{{col1, col2, col3, col4, col5}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int"}, {"float"}, {"int16"}}};

  std::vector<char> out_buffer;
  auto destination     = cudf::io::sink_info(&out_buffer);
  auto options_builder = cudf::io::json_writer_options_builder(destination, tbl_view)
                           .include_nulls(true)
                           .metadata(mt)
                           .lines(false)
                           .na_rep("null");

  cudf::io::write_json(options_builder.build(), rmm::mr::get_current_device_resource());

  std::string const expected =
    R"([{"col1":"a","col2":"d","int":1,"float":1.5,"int16":null},{"col1":"b","col2":"e","int":2,"float":2.5,"int16":2},{"col1":"c","col2":"f","int":3,"float":3.5,"int16":null}])";
  EXPECT_EQ(expected, std::string(out_buffer.data(), out_buffer.size()));
}

TEST_F(JsonWriterTest, SimpleNested)
{
  std::string const data = R"(
{"a": 1, "b": 2, "c": {"d": 3        }, "f": 5.5,  "g": [1]}
{"a": 6, "b": 7, "c": {"d": 8        }, "f": 10.5, "g": null}
{"a": 1, "b": 2, "c": {        "e": 4}, "f": 5.5,  "g": [2, null]}
{"a": 6, "b": 7, "c": {        "e": 9}, "f": 10.5, "g": [3, 4, 5]} )";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{data.data(), data.size()})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  cudf::table_view tbl_view            = result.tbl->view();
  cudf::io::table_metadata mt{result.metadata};

  std::vector<char> out_buffer;
  auto destination     = cudf::io::sink_info(&out_buffer);
  auto options_builder = cudf::io::json_writer_options_builder(destination, tbl_view)
                           .include_nulls(false)
                           .metadata(mt)
                           .lines(true)
                           .na_rep("null");

  cudf::io::write_json(options_builder.build(), rmm::mr::get_current_device_resource());
  std::string const expected = R"({"a":1,"b":2,"c":{"d":3},"f":5.5,"g":[1]}
{"a":6,"b":7,"c":{"d":8},"f":10.5}
{"a":1,"b":2,"c":{"e":4},"f":5.5,"g":[2,null]}
{"a":6,"b":7,"c":{"e":9},"f":10.5,"g":[3,4,5]}
)";
  EXPECT_EQ(expected, std::string(out_buffer.data(), out_buffer.size()));
}

TEST_F(JsonWriterTest, MixedNested)
{
  std::string const data = R"(
{"a": 1, "b": 2, "c": {"d": [3]      }, "f": 5.5,  "g": [ {"h": 1}]}
{"a": 6, "b": 7, "c": {"d": [8]      }, "f": 10.5, "g": null}
{"a": 1, "b": 2, "c": {        "e": 4}, "f": 5.5,  "g": [{"h": 2}, null]}
{"a": 6, "b": 7, "c": {        "e": 9}, "f": 10.5, "g": [{"h": 3}, {"h": 4}, {"h": 5}]} )";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{data.data(), data.size()})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  cudf::table_view tbl_view            = result.tbl->view();
  cudf::io::table_metadata mt{result.metadata};

  std::vector<char> out_buffer;
  auto destination     = cudf::io::sink_info(&out_buffer);
  auto options_builder = cudf::io::json_writer_options_builder(destination, tbl_view)
                           .include_nulls(false)
                           .metadata(mt)
                           .lines(false)
                           .na_rep("null");

  cudf::io::write_json(options_builder.build(), rmm::mr::get_current_device_resource());
  std::string const expected =
    R"([{"a":1,"b":2,"c":{"d":[3]},"f":5.5,"g":[{"h":1}]},)"
    R"({"a":6,"b":7,"c":{"d":[8]},"f":10.5},)"
    R"({"a":1,"b":2,"c":{"e":4},"f":5.5,"g":[{"h":2},null]},)"
    R"({"a":6,"b":7,"c":{"e":9},"f":10.5,"g":[{"h":3},{"h":4},{"h":5}]}])";
  EXPECT_EQ(expected, std::string(out_buffer.data(), out_buffer.size()));
}

TEST_F(JsonWriterTest, WriteReadNested)
{
  using namespace cudf::test::iterators;
  using LCW = cudf::test::lists_column_wrapper<int64_t>;
  cudf::test::fixed_width_column_wrapper<int> a{1, 6, 1, 6};
  cudf::test::fixed_width_column_wrapper<uint8_t> b{2, 7, 2, 7};
  cudf::test::fixed_width_column_wrapper<int64_t> d{{3, 8, 0, 0}, nulls_at({2, 3})};
  cudf::test::fixed_width_column_wrapper<int64_t> e{{0, 0, 4, 9}, nulls_at({0, 1})};
  cudf::test::structs_column_wrapper c{{d, e}};
  cudf::test::fixed_width_column_wrapper<float> f{5.5, 10.5, 5.5, 10.5};
  LCW g{{LCW{1}, LCW{0}, LCW{{2, 0}, null_at(1)}, LCW{3, 4, 5}}, null_at(1)};
  cudf::table_view tbl_view{{a, b, c, f, g}};
  cudf::io::table_metadata mt{{{"a"}, {"b"}, {"c"}, {"f"}, {"g"}}};
  mt.schema_info[2].children = {{"d"}, {"e"}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(false)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .build();

  cudf::io::write_json(out_options, rmm::mr::get_current_device_resource());
  std::string const expected = R"({"a":1,"b":2,"c":{"d":3},"f":5.5,"g":[1]}
{"a":6,"b":7,"c":{"d":8},"f":10.5}
{"a":1,"b":2,"c":{"e":4},"f":5.5,"g":[2,null]}
{"a":6,"b":7,"c":{"e":9},"f":10.5,"g":[3,4,5]}
)";
  auto const output_string   = std::string(out_buffer.data(), out_buffer.size());
  EXPECT_EQ(expected, output_string);

  // Read back the written JSON, and compare with the original table
  // Without type information
  auto in_options = cudf::io::json_reader_options::builder(
                      cudf::io::source_info{output_string.data(), output_string.size()})
                      .lines(true)
                      .build();

  auto result             = cudf::io::read_json(in_options);
  auto tbl_out            = result.tbl->view();
  auto const int64_dtype  = cudf::data_type{cudf::type_id::INT64};
  auto const double_dtype = cudf::data_type{cudf::type_id::FLOAT64};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*cudf::cast(a, int64_dtype), tbl_out.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*cudf::cast(b, int64_dtype), tbl_out.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, tbl_out.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*cudf::cast(f, double_dtype), tbl_out.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(g, tbl_out.column(4));

  mt.schema_info[4].children = {{"offsets"}, {"element"}};  // list child column names
  EXPECT_EQ(mt.schema_info.size(), result.metadata.schema_info.size());
  for (auto i = 0UL; i < mt.schema_info.size(); i++) {
    EXPECT_EQ(mt.schema_info[i].name, result.metadata.schema_info[i].name) << "[" << i << "]";
    EXPECT_EQ(mt.schema_info[i].children.size(), result.metadata.schema_info[i].children.size())
      << "[" << i << "]";
    for (auto j = 0UL; j < mt.schema_info[i].children.size(); j++) {
      EXPECT_EQ(mt.schema_info[i].children[j].name, result.metadata.schema_info[i].children[j].name)
        << "[" << i << "][" << j << "]";
    }
  }

  // Read with type information
  std::map<std::string, cudf::io::schema_element> types;
  types["a"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::INT32}};
  types["b"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::UINT8}};
  types["c"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::STRUCT}};
  types["c"].child_types["d"] = cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}};
  types["c"].child_types["e"] = cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}};
  types["f"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::FLOAT32}};
  types["g"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::LIST}};
  types["g"].child_types["element"] =
    cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}};

  in_options.set_dtypes(types);
  result  = cudf::io::read_json(in_options);
  tbl_out = result.tbl->view();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(a, tbl_out.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(b, tbl_out.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, tbl_out.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(f, tbl_out.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(g, tbl_out.column(4));
  EXPECT_EQ(mt.schema_info.size(), result.metadata.schema_info.size());
  for (auto i = 0UL; i < mt.schema_info.size(); i++) {
    EXPECT_EQ(mt.schema_info[i].name, result.metadata.schema_info[i].name) << "[" << i << "]";
    EXPECT_EQ(mt.schema_info[i].children.size(), result.metadata.schema_info[i].children.size())
      << "[" << i << "]";
    for (auto j = 0UL; j < mt.schema_info[i].children.size(); j++) {
      EXPECT_EQ(mt.schema_info[i].children[j].name, result.metadata.schema_info[i].children[j].name)
        << "[" << i << "][" << j << "]";
    }
  }

  // Without children column names
  mt.schema_info[2].children.clear();
  out_options.set_metadata(mt);
  out_buffer.clear();
  cudf::io::write_json(out_options, rmm::mr::get_current_device_resource());

  in_options = cudf::io::json_reader_options::builder(
                 cudf::io::source_info{out_buffer.data(), out_buffer.size()})
                 .lines(true)
                 .build();
  result = cudf::io::read_json(in_options);

  mt.schema_info[2].children = {{"0"}, {"1"}};
  EXPECT_EQ(mt.schema_info.size(), result.metadata.schema_info.size());
  for (auto i = 0UL; i < mt.schema_info.size(); i++) {
    EXPECT_EQ(mt.schema_info[i].name, result.metadata.schema_info[i].name) << "[" << i << "]";
    EXPECT_EQ(mt.schema_info[i].children.size(), result.metadata.schema_info[i].children.size())
      << "[" << i << "]";
    for (auto j = 0UL; j < mt.schema_info[i].children.size(); j++) {
      EXPECT_EQ(mt.schema_info[i].children[j].name, result.metadata.schema_info[i].children[j].name)
        << "[" << i << "][" << j << "]";
    }
  }

  // without column names
  out_options.set_metadata(cudf::io::table_metadata{});
  out_buffer.clear();
  cudf::io::write_json(out_options, rmm::mr::get_current_device_resource());
  in_options = cudf::io::json_reader_options::builder(
                 cudf::io::source_info{out_buffer.data(), out_buffer.size()})
                 .lines(true)
                 .build();
  result = cudf::io::read_json(in_options);

  mt.schema_info             = {{"0"}, {"1"}, {"2"}, {"3"}, {"4"}};
  mt.schema_info[2].children = {{"0"}, {"1"}};
  mt.schema_info[4].children = {{"offsets"}, {"element"}};  // list child column names
  EXPECT_EQ(mt.schema_info.size(), result.metadata.schema_info.size());
  for (auto i = 0UL; i < mt.schema_info.size(); i++) {
    EXPECT_EQ(mt.schema_info[i].name, result.metadata.schema_info[i].name) << "[" << i << "]";
    EXPECT_EQ(mt.schema_info[i].children.size(), result.metadata.schema_info[i].children.size())
      << "[" << i << "]";
    for (auto j = 0UL; j < mt.schema_info[i].children.size(); j++) {
      EXPECT_EQ(mt.schema_info[i].children[j].name, result.metadata.schema_info[i].children[j].name)
        << "[" << i << "][" << j << "]";
    }
  }
}

TEST_F(JsonWriterTest, SpecialChars)
{
  cudf::test::fixed_width_column_wrapper<int> a{1, 6, 1, 6};
  cudf::test::strings_column_wrapper b{"abcd", "b\b\f\n\r\t", "\"c\"", "/\\"};
  cudf::table_view tbl_view{{a, b}};
  cudf::io::table_metadata mt{{{"\"a\""}, {"\'b\'"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(false)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .build();

  cudf::io::write_json(out_options, rmm::mr::get_current_device_resource());
  std::string const expected = R"({"\"a\"":1,"'b'":"abcd"}
{"\"a\"":6,"'b'":"b\b\f\n\r\t"}
{"\"a\"":1,"'b'":"\"c\""}
{"\"a\"":6,"'b'":"\/\\"}
)";
  auto const output_string   = std::string(out_buffer.data(), out_buffer.size());
  EXPECT_EQ(expected, output_string);
}

CUDF_TEST_PROGRAM_MAIN()
