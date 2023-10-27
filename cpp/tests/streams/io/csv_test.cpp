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

#include <cudf/io/csv.hpp>
#include <cudf/io/detail/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <random>
#include <sstream>
#include <string>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class CSVTest : public cudf::test::BaseFixture {};

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

TEST_F(CSVTest, CSVReader)
{
  constexpr auto num_rows = 10;
  auto int8_values        = random_values<int8_t>(num_rows);
  auto int16_values       = random_values<int16_t>(num_rows);
  auto int32_values       = random_values<int32_t>(num_rows);
  auto int64_values       = random_values<int64_t>(num_rows);
  auto uint8_values       = random_values<uint8_t>(num_rows);
  auto uint16_values      = random_values<uint16_t>(num_rows);
  auto uint32_values      = random_values<uint32_t>(num_rows);
  auto uint64_values      = random_values<uint64_t>(num_rows);
  auto float32_values     = random_values<float>(num_rows);
  auto float64_values     = random_values<double>(num_rows);

  auto filepath = temp_env->get_temp_dir() + "MultiColumn.csv";
  {
    std::ostringstream line;
    for (int i = 0; i < num_rows; ++i) {
      line << std::to_string(int8_values[i]) << "," << int16_values[i] << "," << int32_values[i]
           << "," << int64_values[i] << "," << std::to_string(uint8_values[i]) << ","
           << uint16_values[i] << "," << uint32_values[i] << "," << uint64_values[i] << ","
           << float32_values[i] << "," << float64_values[i] << "\n";
    }
    std::ofstream outfile(filepath, std::ofstream::out);
    outfile << line.str();
  }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{filepath})
      .header(-1)
      .dtypes({cudf::data_type{cudf::type_id::INT8},
               cudf::data_type{cudf::type_id::INT16},
               cudf::data_type{cudf::type_id::INT32},
               cudf::data_type{cudf::type_id::INT64},
               cudf::data_type{cudf::type_id::UINT8},
               cudf::data_type{cudf::type_id::UINT16},
               cudf::data_type{cudf::type_id::UINT32},
               cudf::data_type{cudf::type_id::UINT64},
               cudf::data_type{cudf::type_id::FLOAT32},
               cudf::data_type{cudf::type_id::FLOAT64}});
  auto result = cudf::io::read_csv(in_opts, cudf::test::get_default_stream());
}

TEST_F(CSVTest, CSVWriter)
{
  auto const input_strings = cudf::test::strings_column_wrapper{
    std::string{"All"} + "," + "the" + "," + "leaves", "are\"brown", "and\nthe\nsky\nis\ngrey"};
  auto const input_table = cudf::table_view{{input_strings}};

  auto const filepath = temp_env->get_temp_dir() + "unquoted.csv";
  auto w_options = cudf::io::csv_writer_options::builder(cudf::io::sink_info{filepath}, input_table)
                     .include_header(false)
                     .inter_column_delimiter(',')
                     .quoting(cudf::io::quote_style::NONE);
  cudf::io::write_csv(w_options.build(), cudf::test::get_default_stream());
}
