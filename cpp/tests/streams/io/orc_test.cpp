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

template <typename T>
inline auto random_values(size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same_v<T1, bool>,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point_v<T1>,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

inline auto random_strings(size_t size)
{
  std::vector<std::string> values(size);
  std::vector<int> pos(size);
  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<> dist(0, 25);
  std::transform(values.begin(), values.end(), values.begin(), [&pos](std::string s) {
    std::generate(pos.begin(), pos.end(), [&]() { return dist(engine); });
    std::transform(
      pos.begin(), pos.end(), std::back_inserter(s), [](const int& p) { return p + 'a'; });
    return s;
  });
  return values;
}

TEST_F(ORCTest, ORCWriter)
{
  constexpr auto num_rows = 10;

  auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int64_t>(num_rows);
  auto col6_data = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    return numeric::decimal128{col6_vals[i], numeric::scale_type{12}};
  });
  auto col7_data = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    return numeric::decimal128{col6_vals[i], numeric::scale_type{-12}};
  });

  cudf::test::fixed_width_column_wrapper<bool> col0(col0_data.begin(), col0_data.end());
  cudf::test::fixed_width_column_wrapper<int8_t> col1(col1_data.begin(), col1_data.end());
  cudf::test::fixed_width_column_wrapper<int16_t> col2(col2_data.begin(), col2_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(col3_data.begin(), col3_data.end());
  cudf::test::fixed_width_column_wrapper<float> col4(col4_data.begin(), col4_data.end());
  cudf::test::fixed_width_column_wrapper<double> col5(col5_data.begin(), col5_data.end());
  cudf::test::fixed_width_column_wrapper<numeric::decimal128> col6(col6_data, col6_data + num_rows);
  cudf::test::fixed_width_column_wrapper<numeric::decimal128> col7(col7_data, col7_data + num_rows);

  cudf::test::lists_column_wrapper<int64_t> col8{
    {9, 8}, {7, 6, 5}, {}, {4}, {3, 2, 1, 0}, {20, 21, 22, 23, 24}, {}, {66, 666}, {}, {-1, -2}};

  cudf::test::fixed_width_column_wrapper<int32_t> child_col{
    48, 27, 25, 31, 351, 351, 29, 15, -1, -99};
  cudf::test::structs_column_wrapper col9{child_col};

  auto col10_data = random_strings(num_rows);
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

  auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int64_t>(num_rows);
  auto col6_data = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    return numeric::decimal128{col6_vals[i], numeric::scale_type{12}};
  });
  auto col7_data = cudf::detail::make_counting_transform_iterator(0, [&](auto i) {
    return numeric::decimal128{col6_vals[i], numeric::scale_type{-12}};
  });

  cudf::test::fixed_width_column_wrapper<bool> col0(col0_data.begin(), col0_data.end());
  cudf::test::fixed_width_column_wrapper<int8_t> col1(col1_data.begin(), col1_data.end());
  cudf::test::fixed_width_column_wrapper<int16_t> col2(col2_data.begin(), col2_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(col3_data.begin(), col3_data.end());
  cudf::test::fixed_width_column_wrapper<float> col4(col4_data.begin(), col4_data.end());
  cudf::test::fixed_width_column_wrapper<double> col5(col5_data.begin(), col5_data.end());
  cudf::test::fixed_width_column_wrapper<numeric::decimal128> col6(col6_data, col6_data + num_rows);
  cudf::test::fixed_width_column_wrapper<numeric::decimal128> col7(col7_data, col7_data + num_rows);

  cudf::test::lists_column_wrapper<int64_t> col8{
    {9, 8}, {7, 6, 5}, {}, {4}, {3, 2, 1, 0}, {20, 21, 22, 23, 24}, {}, {66, 666}, {}, {-1, -2}};

  cudf::test::fixed_width_column_wrapper<int32_t> child_col{
    48, 27, 25, 31, 351, 351, 29, 15, -1, -99};
  cudf::test::structs_column_wrapper col9{child_col};

  auto col10_data = random_strings(num_rows);
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
