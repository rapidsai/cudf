/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <src/io/comp/nvcomp_adapter.hpp>

#include <type_traits>

template <typename T, typename SourceElementT = T>
using column_wrapper =
  typename std::conditional<std::is_same_v<T, cudf::string_view>,
                            cudf::test::strings_column_wrapper,
                            cudf::test::fixed_width_column_wrapper<T, SourceElementT>>::type;

using str_col     = column_wrapper<cudf::string_view>;
using bool_col    = column_wrapper<bool>;
using int8_col    = column_wrapper<int8_t>;
using int16_col   = column_wrapper<int16_t>;
using int32_col   = column_wrapper<int32_t>;
using int64_col   = column_wrapper<int64_t>;
using float32_col = column_wrapper<float>;
using float64_col = column_wrapper<double>;
using dec32_col   = column_wrapper<numeric::decimal32>;
using dec64_col   = column_wrapper<numeric::decimal64>;
using dec128_col  = column_wrapper<numeric::decimal128>;
using struct_col  = cudf::test::structs_column_wrapper;
template <typename T>
using list_col = cudf::test::lists_column_wrapper<T>;

using column     = cudf::column;
using table      = cudf::table;
using table_view = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

template <typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns,
                                                       cudf::size_type num_rows,
                                                       bool include_validity)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
  std::vector<column_wrapper<T>> src_cols(num_columns);
  for (int idx = 0; idx < num_columns; idx++) {
    auto rand_elements =
      cudf::detail::make_counting_transform_iterator(0, [](T i) { return rand(); });
    if (include_validity) {
      src_cols[idx] = column_wrapper<T>(rand_elements, rand_elements + num_rows, valids);
    } else {
      src_cols[idx] = column_wrapper<T>(rand_elements, rand_elements + num_rows);
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
  std::transform(src_cols.begin(), src_cols.end(), columns.begin(), [](column_wrapper<T>& in) {
    auto ret                    = in.release();
    [[maybe_unused]] auto nulls = ret->has_nulls();  // pre-cache the null count
    return ret;
  });
  return std::make_unique<cudf::table>(std::move(columns));
}

namespace {
// Generates a vector of uniform random values of type T
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
}  // namespace

// Base test fixture for tests
struct OrcWriterTest : public cudf::test::BaseFixture {};

struct OrcWriterTestStripes
  : public OrcWriterTest,
    public ::testing::WithParamInterface<std::tuple<size_t, cudf::size_type>> {};

TEST_F(OrcWriterTestStripes, StripeSize)
{
  constexpr auto num_rows = 1000000;
  // auto const [size_bytes, size_rows] = GetParam();

  auto const seq_col = random_values<int>(num_rows);
  auto const validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
  column_wrapper<int64_t> col{seq_col.begin(), seq_col.end(), validity};

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());

  // printf("input col: \n");
  // cudf::test::print(cols.front()->view());

  auto const expected = std::make_unique<table>(std::move(cols));

  auto validate = [&](std::vector<char> const& orc_buffer) {
    // auto const expected_stripe_num = 6;
    // auto const stats               = cudf::io::read_parsed_orc_statistics(
    //   cudf::io::source_info(orc_buffer.data(), orc_buffer.size()));
    // EXPECT_EQ(stats.stripes_stats.size(), expected_stripe_num);

    cudf::io::orc_reader_options in_opts =
      cudf::io::orc_reader_options::builder(
        cudf::io::source_info(orc_buffer.data(), orc_buffer.size()))
        .use_index(false);
    auto result = cudf::io::read_orc(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), result.tbl->view());
  };

  {
    std::vector<char> out_buffer_chunked;
    cudf::io::chunked_orc_writer_options opts =
      cudf::io::chunked_orc_writer_options::builder(cudf::io::sink_info(&out_buffer_chunked))
        .stripe_size_rows(10000);
    cudf::io::orc_chunked_writer(opts).write(expected->view());

    validate(out_buffer_chunked);
  }
}

// INSTANTIATE_TEST_CASE_P(OrcWriterTest,
//                         OrcWriterTestStripes,
//                         ::testing::Values(std::make_tuple(800000ul, 1000000)));

// INSTANTIATE_TEST_CASE_P(OrcWriterTest,
//                         OrcWriterTestStripes,
//                         ::testing::Values(std::make_tuple(800000ul, 1000000),
//                                           std::make_tuple(2000000ul, 1000000),
//                                           std::make_tuple(4000000ul, 1000000),
//                                           std::make_tuple(8000000ul, 1000000),
//                                           std::make_tuple(8000000ul, 500000),
//                                           std::make_tuple(8000000ul, 250000),
//                                           std::make_tuple(8000000ul, 100000)));
