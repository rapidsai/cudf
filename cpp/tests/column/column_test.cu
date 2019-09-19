/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.cuh>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_list.hpp>
#include <tests/utilities/typed_tests.hpp>

#include <gmock/gmock.h>

struct ColumnTest : public cudf::test::BaseFixture {};

template <typename T>
class TypedColumnTest : public ColumnTest {};

TYPED_TEST_CASE(TypedColumnTest, cudf::test::AllTypes);

TYPED_TEST(TypedColumnTest, First) {
  constexpr cudf::size_type size{1000};
  rmm::device_buffer data{size};
  cudf::column col{cudf::data_type{cudf::exp::type_to_id<TypeParam>()}, 100,
                   data};

  auto d_col = cudf::column_device_view::create(col);

  auto d_table = cudf::table_device_view::create(cudf::table_view{{col}});

  cudf::test::expect_columns_equal(col, col);

  CUDA_TRY(cudaGetLastError());
}