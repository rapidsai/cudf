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

#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/io/orc_types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class ORCTest : public cudf::test::BaseFixture {};

TEST_F(ORCTest, ORCWriter)
{
  constexpr auto num_rows = 10;

  std::vector<size_t> zeros(num_rows, 0);
  std::vector<size_t> ones(num_rows, 1);
  auto col6_data = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    return numeric::decimal128{ones[i], numeric::scale_type{12}};
  });
  auto col7_data = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    return numeric::decimal128{ones[i], numeric::scale_type{-12}};
  });

  cudf::test::fixed_width_column_wrapper<bool> col0(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int8_t> col1(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int16_t> col2(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<float> col4(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<double> col5(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<numeric::decimal128> col6(col6_data, col6_data + num_rows);
  cudf::test::fixed_width_column_wrapper<numeric::decimal128> col7(col7_data, col7_data + num_rows);

  cudf::test::lists_column_wrapper<int64_t> col8{
    {1, 1}, {1, 1, 1}, {}, {1}, {1, 1, 1, 1}, {1, 1, 1, 1, 1}, {}, {1, -1}, {}, {-1, -1}};

  cudf::test::fixed_width_column_wrapper<int32_t> child_col(ones.begin(), ones.end());
  cudf::test::structs_column_wrapper col9{child_col};

  std::vector<std::string> col10_data(num_rows, "rapids");
  cudf::test::strings_column_wrapper col10(col10_data.begin(), col10_data.end());

  cudf::table_view tab({col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10});

  cudf::io::table_input_metadata tab_metadata(tab);
  tab_metadata.column_metadata[0].set_name("bools");
  tab_metadata.column_metadata[1].set_name("int8s");
  tab_metadata.column_metadata[2].set_name("int16s");
  tab_metadata.column_metadata[3].set_name("int32s");
  tab_metadata.column_metadata[4].set_name("floats");
  tab_metadata.column_metadata[5].set_name("doubles");
  tab_metadata.column_metadata[6].set_name("decimal_pos_scale");
  tab_metadata.column_metadata[7].set_name("decimal_neg_scale");
  tab_metadata.column_metadata[8].set_name("lists");
  tab_metadata.column_metadata[9].set_name("structs");
  tab_metadata.column_metadata[10].set_name("strings");

  auto filepath = temp_env->get_temp_filepath("OrcMultiColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, tab)
      .metadata(tab_metadata);
  cudf::io::write_orc(out_opts, cudf::test::get_default_stream());
}

TEST_F(ORCTest, ORCReader)
{
  constexpr auto num_rows = 10;

  std::vector<size_t> zeros(num_rows, 0);
  std::vector<size_t> ones(num_rows, 1);
  auto col6_data = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    return numeric::decimal128{ones[i], numeric::scale_type{12}};
  });
  auto col7_data = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    return numeric::decimal128{ones[i], numeric::scale_type{-12}};
  });

  cudf::test::fixed_width_column_wrapper<bool> col0(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int8_t> col1(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int16_t> col2(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<float> col4(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<double> col5(zeros.begin(), zeros.end());
  cudf::test::fixed_width_column_wrapper<numeric::decimal128> col6(col6_data, col6_data + num_rows);
  cudf::test::fixed_width_column_wrapper<numeric::decimal128> col7(col7_data, col7_data + num_rows);

  cudf::test::lists_column_wrapper<int64_t> col8{
    {1, 1}, {1, 1, 1}, {}, {1}, {1, 1, 1, 1}, {1, 1, 1, 1, 1}, {}, {1, -1}, {}, {-1, -1}};

  cudf::test::fixed_width_column_wrapper<int32_t> child_col(ones.begin(), ones.end());
  cudf::test::structs_column_wrapper col9{child_col};

  std::vector<std::string> col10_data(num_rows, "rapids");
  cudf::test::strings_column_wrapper col10(col10_data.begin(), col10_data.end());

  cudf::table_view tab({col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10});

  cudf::io::table_input_metadata tab_metadata(tab);
  tab_metadata.column_metadata[0].set_name("bools");
  tab_metadata.column_metadata[1].set_name("int8s");
  tab_metadata.column_metadata[2].set_name("int16s");
  tab_metadata.column_metadata[3].set_name("int32s");
  tab_metadata.column_metadata[4].set_name("floats");
  tab_metadata.column_metadata[5].set_name("doubles");
  tab_metadata.column_metadata[6].set_name("decimal_pos_scale");
  tab_metadata.column_metadata[7].set_name("decimal_neg_scale");
  tab_metadata.column_metadata[8].set_name("lists");
  tab_metadata.column_metadata[9].set_name("structs");
  tab_metadata.column_metadata[10].set_name("strings");

  auto filepath = temp_env->get_temp_filepath("OrcMultiColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, tab)
      .metadata(tab_metadata);
  cudf::io::write_orc(out_opts, cudf::test::get_default_stream());

  cudf::io::orc_reader_options read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{{filepath}});
  auto result = cudf::io::read_orc(read_opts, cudf::test::get_default_stream());

  auto meta        = read_orc_metadata(cudf::io::source_info{filepath});
  auto const stats = cudf::io::read_parsed_orc_statistics(cudf::io::source_info{filepath});
}
