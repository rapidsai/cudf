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

#include <rmm/device_uvector.hpp>

#include <string>
#include <vector>

struct JsonWriterTest : public cudf::test::BaseFixture {
};

TEST_F(JsonWriterTest, PlainTable)
{
  cudf::test::strings_column_wrapper col1{"a", "b", "c"};
  cudf::test::strings_column_wrapper col2{"d", "e", "f"};
  cudf::test::fixed_width_column_wrapper<int> col3{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> col4{1.5, 2.5, 3.5};
  cudf::test::fixed_width_column_wrapper<int16_t> col5{{1, 2, 3},
                                                       cudf::test::iterators::nulls_at({0, 2})};
  cudf::table_view tbl_view{{col1, col2, col3, col4, col5}};
  std::vector<std::string> column_names{"col1", "col2", "int", "float", "int16"};
  cudf::io::table_metadata mt{column_names};

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

CUDF_TEST_PROGRAM_MAIN()
