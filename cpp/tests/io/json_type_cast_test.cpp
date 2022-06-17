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
#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <src/io/json/data_casting.cuh>

#include <type_traits>

struct JSONTypeCastTest : public cudf::test::BaseFixture {
};

TEST_F(JSONTypeCastTest, Basic)
{
  std::vector<cudf::data_type> types{cudf::data_type{cudf::type_id::INT32}};
  cudf::io::json::experimental::parse_data<cudf::string_view**>(
    {}, types, rmm::cuda_stream_default);
  EXPECT_TRUE(0);
}

CUDF_TEST_PROGRAM_MAIN()
