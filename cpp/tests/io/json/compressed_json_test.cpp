/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "io/comp/comp.hpp"
#include "io/comp/io_uncomp.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/io/json.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <fstream>
#include <limits>
#include <memory>
#include <type_traits>

#define wrapper cudf::test::fixed_width_column_wrapper
using float_wrapper        = wrapper<float>;
using float64_wrapper      = wrapper<double>;
using int_wrapper          = wrapper<int>;
using int8_wrapper         = wrapper<int8_t>;
using int16_wrapper        = wrapper<int16_t>;
using int64_wrapper        = wrapper<int64_t>;
using timestamp_ms_wrapper = wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>;
using bool_wrapper         = wrapper<bool>;

using cudf::data_type;
using cudf::type_to_id;

template <typename T>
auto dtype()
{
  return data_type{type_to_id<T>()};
}

template <typename T, typename SourceElementT = T>
using column_wrapper =
  std::conditional_t<std::is_same_v<T, cudf::string_view>,
                     cudf::test::strings_column_wrapper,
                     cudf::test::fixed_width_column_wrapper<T, SourceElementT>>;

cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

/**
 * @brief Test fixture for parametrized JSON reader tests
 */
struct JsonCompressedIOTest : public cudf::test::BaseFixture,
                              public testing::WithParamInterface<cudf::io::compression_type> {};

// Parametrize qualifying JSON tests for multiple compression types
INSTANTIATE_TEST_SUITE_P(JsonCompressedIOTest,
                         JsonCompressedIOTest,
                         ::testing::Values(cudf::io::compression_type::GZIP,
                                           cudf::io::compression_type::SNAPPY,
                                           cudf::io::compression_type::NONE));

/**
 * @brief Generates a JSON lines string that uses the record orient
 *
 * @param records An array of a map of key-value pairs
 * @param record_delimiter The delimiter to be used to delimit a record
 * @param prefix The prefix prepended to the whole string
 * @param suffix The suffix to be appended after the whole string
 * @return The JSON lines string that uses the record orient
 */
std::string to_records_orient(std::vector<std::map<std::string, std::string>> const& records,
                              std::string record_delimiter,
                              std::string prefix = "",
                              std::string suffix = "")
{
  std::string result = prefix;
  for (auto record_it = std::cbegin(records); record_it != std::cend(records); record_it++) {
    result += "{";
    for (auto kv_pair_it = std::cbegin(*record_it); kv_pair_it != std::cend(*record_it);
         kv_pair_it++) {
      auto const& [key, value] = *kv_pair_it;
      result += "\"" + key + "\":" + value;
      result += (kv_pair_it != std::prev(std::end(*record_it))) ? ", " : "";
    }
    result += "}";
    if (record_it != std::prev(std::end(records))) { result += record_delimiter; }
  }
  return (result + suffix);
}

TEST_P(JsonCompressedIOTest, BasicJsonLines)
{
  cudf::io::compression_type const comptype = GetParam();
  rmm::cuda_stream stream{};
  rmm::cuda_stream_view stream_view(stream);
  std::string data = to_records_orient(
    {{{"0", "1"}, {"1", "1.1"}}, {{"0", "2"}, {"1", "2.2"}}, {{"0", "3"}, {"1", "3.3"}}}, "\n");

  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(data.data()), data.size()),
      stream_view);
    auto decomp_out_buffer =
      cudf::io::decompress(comptype, cudf::host_span<uint8_t const>(cdata.data(), cdata.size()));
    std::string const expected = R"({"0":1, "1":1.1}
{"0":2, "1":2.2}
{"0":3, "1":3.3})";
    EXPECT_EQ(
      expected,
      std::string(reinterpret_cast<char*>(decomp_out_buffer.data()), decomp_out_buffer.size()));
  } else
    cdata = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(data.data()),
                                 reinterpret_cast<uint8_t*>(data.data()) + data.size());

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<uint8_t>(cdata.data(), cdata.size())})
      .dtypes(std::vector<data_type>{dtype<int32_t>(), dtype<double>()})
      .compression(comptype)
      .lines(true);
  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 3);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::type_id::FLOAT64);

  EXPECT_EQ(result.metadata.schema_info[0].name, "0");
  EXPECT_EQ(result.metadata.schema_info[1].name, "1");

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), int_wrapper{{1, 2, 3}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2, 3.3}});
}
