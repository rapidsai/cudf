/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>

#include <bits/stdint-intn.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/maps/map_lookup.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <rmm/device_buffer.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>
#include "cudf/scalar/scalar.hpp"

using vector_of_columns = std::vector<std::unique_ptr<cudf::column>>;
using cudf::size_type;

struct MapLookupTest : public cudf::test::BaseFixture {
};

TEST_F(MapLookupTest, Basics)
{
  using namespace cudf::test;

  auto keys   = strings_column_wrapper{"0", "1", "1", "2", "2", "3"};
  auto values = strings_column_wrapper{"00", "11", "11", "22", "22", "33"};
  auto pairs  = structs_column_wrapper{{keys, values}}.release();

  auto maps = cudf::make_lists_column(3,
                                      fixed_width_column_wrapper<size_type>{{0, 2, 4, 6}}.release(),
                                      std::move(pairs),
                                      cudf::UNKNOWN_NULL_COUNT,
                                      {});

  auto lookup = cudf::map_lookup(maps->view(), cudf::string_scalar("1"));

  expect_columns_equivalent(lookup->view(), strings_column_wrapper{{"11", "11", ""}, {1, 1, 0}});
}

TEST_F(MapLookupTest, EmptyMaps) { using namespace cudf::test; }

TEST_F(MapLookupTest, NullMaps) { using namespace cudf::test; }

TEST_F(MapLookupTest, DefensiveChecks)
{
  using namespace cudf::test;

  auto list_offsets = fixed_width_column_wrapper<size_type>{{0, 2, 4}};
  auto string_keys  = strings_column_wrapper{"0", "1", "1", "2"};
  auto int_values   = fixed_width_column_wrapper<int32_t>{0, 1, 1, 2};

  // Check that API insists on receiving List<Struct<String,String>>.
  EXPECT_THROW(cudf::map_lookup(list_offsets, cudf::string_scalar("foo!")), cudf::logic_error);

  auto structs = structs_column_wrapper{{string_keys, int_values}};
  EXPECT_THROW(cudf::map_lookup(structs, cudf::string_scalar("foo!")), cudf::logic_error);

  auto list_of_structs = cudf::make_lists_column(
    2, list_offsets.release(), structs.release(), cudf::UNKNOWN_NULL_COUNT, {});

  EXPECT_THROW(cudf::map_lookup(list_of_structs->view(), cudf::string_scalar("foo!")),
               cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
