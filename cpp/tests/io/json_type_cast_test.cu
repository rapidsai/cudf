/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/data_casting.cuh>
#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <type_traits>

using namespace cudf::test::iterators;

struct JSONTypeCastTest : public cudf::test::BaseFixture {
};

namespace {
struct to_thrust_pair_fn {
  __device__ thrust::pair<const char*, cudf::size_type> operator()(
    thrust::pair<cudf::string_view, bool> const& p)
  {
    return {p.first.data(), p.first.size_bytes()};
  }
};
}  // namespace

auto default_json_options()
{
  auto parse_opts = cudf::io::parse_options{',', '\n', '\"', '.'};

  auto const stream     = cudf::default_stream_value;
  parse_opts.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  parse_opts.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  parse_opts.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);
  return parse_opts;
}

TEST_F(JSONTypeCastTest, String)
{
  auto const stream = cudf::default_stream_value;
  auto mr           = rmm::mr::get_current_device_resource();
  auto const type   = cudf::data_type{cudf::type_id::STRING};

  auto in_valids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 4; });
  std::vector<const char*> input_values{"this", "is", "null", "of", "", "strings", R"("null")"};
  cudf::test::strings_column_wrapper input(input_values.begin(), input_values.end(), in_valids);

  auto d_column = cudf::column_device_view::create(input);
  rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> svs(d_column->size(), stream);
  thrust::transform(rmm::exec_policy(cudf::default_stream_value),
                    d_column->pair_begin<cudf::string_view, false>(),
                    d_column->pair_end<cudf::string_view, false>(),
                    svs.begin(),
                    to_thrust_pair_fn{});

  auto null_mask_it = no_nulls();
  auto null_mask =
    cudf::test::detail::make_null_mask(null_mask_it, null_mask_it + d_column->size());

  auto str_col = cudf::io::json::experimental::detail::parse_data(
    svs.data(), svs.size(), type, std::move(null_mask), default_json_options().view(), stream, mr);

  auto out_valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 2 and i != 4; });
  std::vector<const char*> expected_values{"this", "is", "", "of", "", "strings", "null"};
  cudf::test::strings_column_wrapper expected(
    expected_values.begin(), expected_values.end(), out_valids);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(str_col->view(), expected);
}

TEST_F(JSONTypeCastTest, Int)
{
  auto const stream = cudf::default_stream_value;
  auto mr           = rmm::mr::get_current_device_resource();
  auto const type   = cudf::data_type{cudf::type_id::INT64};

  cudf::test::strings_column_wrapper data({"1", "null", "3", "true", "5", "false"});
  auto d_column = cudf::column_device_view::create(data);
  rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> svs(d_column->size(), stream);
  thrust::transform(rmm::exec_policy(cudf::default_stream_value),
                    d_column->pair_begin<cudf::string_view, false>(),
                    d_column->pair_end<cudf::string_view, false>(),
                    svs.begin(),
                    to_thrust_pair_fn{});

  auto null_mask_it = no_nulls();
  auto null_mask =
    cudf::test::detail::make_null_mask(null_mask_it, null_mask_it + d_column->size());

  auto col = cudf::io::json::experimental::detail::parse_data(
    svs.data(), svs.size(), type, std::move(null_mask), default_json_options().view(), stream, mr);

  auto expected =
    cudf::test::fixed_width_column_wrapper<int64_t>{{1, 2, 3, 1, 5, 0}, {1, 0, 1, 1, 1, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(col->view(), expected);
}

TEST_F(JSONTypeCastTest, StringEscapes)
{
  auto const stream = cudf::default_stream_value;
  auto mr           = rmm::mr::get_current_device_resource();
  auto const type   = cudf::data_type{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper data({
    R"("\uD83D\uDE80")",
    R"("\uff21\ud83d\ude80\uff21\uff21")",
    R"("invalid char being escaped escape char\-")",
    R"("too few hex digits \u12")",
    R"("too few hex digits for surrogate pair \uD83D\uDE")",
    R"("\u005C")",
    R"("\u27A9")",
    R"("escape with nothing to escape \")",
    R"("\"\\\/\b\f\n\r\t")",
  });
  auto d_column = cudf::column_device_view::create(data);
  rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> svs(d_column->size(), stream);
  thrust::transform(rmm::exec_policy(cudf::default_stream_value),
                    d_column->pair_begin<cudf::string_view, false>(),
                    d_column->pair_end<cudf::string_view, false>(),
                    svs.begin(),
                    to_thrust_pair_fn{});

  auto null_mask_it = no_nulls();
  auto null_mask =
    cudf::test::detail::make_null_mask(null_mask_it, null_mask_it + d_column->size());

  auto col = cudf::io::json::experimental::detail::parse_data(
    svs.data(), svs.size(), type, std::move(null_mask), default_json_options().view(), stream, mr);

  auto expected = cudf::test::strings_column_wrapper{
    {"🚀", "Ａ🚀ＡＡ", "", "", "", "\\", "➩", "", "\"\\/\b\f\n\r\t"},
    {true, true, false, false, false, true, true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(col->view(), expected);
}

CUDF_TEST_PROGRAM_MAIN()
