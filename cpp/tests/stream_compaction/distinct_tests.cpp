/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <cmath>

using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;

struct Distinct : public cudf::test::BaseFixture {
};

TEST_F(Distinct, StringKeyColumn)
{
  // clang-format off
  cudf::test::fixed_width_column_wrapper<int32_t> col    {0, 1, 2, 3, 4, 5, 6};
  cudf::test::fixed_width_column_wrapper<int32_t> key_col{5, 4, 4, 5, 5, 8, 1};
  cudf::table_view input{{col, key_col}};
  std::vector<cudf::size_type> keys{1};

  auto r= cudf::distinct(input, keys);
  cudf::test::print(r->get_column(0).view());
  exit(0);

}
