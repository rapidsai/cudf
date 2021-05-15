/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar_factories.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

namespace cudf {
namespace test {

template <typename T>
class TypedScatterListScalarTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(TypedScatterListScalarTest, FixedWidthTypes);

TYPED_TEST(TypedScatterListScalarTest, FixedWidthScalar)
{
  using LCW  = lists_column_wrapper<TypeParam, int32_t>;
  using FCW  = fixed_width_column_wrapper<TypeParam>;
  using SM_t = fixed_width_column_wrapper<size_type>;

  auto slr = std::make_unique<list_scalar>(FCW{2, 2, 2}, true);
  LCW col{LCW{1, 1, 1}, LCW{8, 8}, LCW{10, 10, 10, 10}, LCW{5}};
  SM_t scatter_map{2, 0};

  LCW expected{LCW{2, 2, 2}, LCW{8, 8}, LCW{2, 2, 2}, LCW{5}};
}

}  // namespace test
}  // namespace cudf
