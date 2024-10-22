/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/io/json.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <string>
#include <vector>

class JSONTest : public cudf::test::BaseFixture {};

TEST_F(JSONTest, JSONreader)
{
  std::string data = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{data.data(), data.size()})
      .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::FLOAT64}})
      .lines(true);
  cudf::io::table_with_metadata result =
    cudf::io::read_json(in_options, cudf::test::get_default_stream());
}

TEST_F(JSONTest, JSONwriter)
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

  cudf::io::write_json(options_builder.build(), cudf::test::get_default_stream());
}
