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
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <src/io/json/data_casting.cuh>

#include <type_traits>

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

TEST_F(JSONTypeCastTest, String)
{
  auto const stream = rmm::cuda_stream_default;
  auto mr           = rmm::mr::get_current_device_resource();
  auto const type   = cudf::data_type{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper data({"this", "is", "a", "column", "of", "strings"});
  auto d_column = cudf::column_device_view::create(data);
  rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> svs(d_column->size(), stream);
  thrust::transform(thrust::device,
                    d_column->pair_begin<cudf::string_view, false>(),
                    d_column->pair_end<cudf::string_view, false>(),
                    svs.begin(),
                    to_thrust_pair_fn{});

  auto str_col = cudf::io::json::experimental::parse_data(
    svs.data(), svs.size(), type, rmm::device_buffer{0, stream}, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(str_col->view(), data);
}

TEST_F(JSONTypeCastTest, Int)
{
  auto const stream = rmm::cuda_stream_default;
  auto mr           = rmm::mr::get_current_device_resource();
  auto const type   = cudf::data_type{cudf::type_id::INT32};

  cudf::test::strings_column_wrapper data({"1", "2", "3", "4", "5", "6"});
  auto d_column = cudf::column_device_view::create(data);
  rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> svs(d_column->size(), stream);
  thrust::transform(thrust::device,
                    d_column->pair_begin<cudf::string_view, false>(),
                    d_column->pair_end<cudf::string_view, false>(),
                    svs.begin(),
                    to_thrust_pair_fn{});

  auto col = cudf::io::json::experimental::parse_data(
    svs.data(), svs.size(), type, rmm::device_buffer{0, stream}, stream, mr);

  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4, 5, 6}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(col->view(), expected);
}

CUDF_TEST_PROGRAM_MAIN()
