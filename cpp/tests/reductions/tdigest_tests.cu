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

#include <cudf/reduction.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/tdigest_utilities.cuh>
#include <cudf_test/type_lists.hpp>

namespace cudf {
namespace test {

template <typename T>
struct ReductionTDigestAllTypes : public cudf::test::BaseFixture {
};
TYPED_TEST_SUITE(ReductionTDigestAllTypes, cudf::test::NumericTypes);

struct reduce_op {
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& values, int delta) const
  {
    // result is a scalar, but we want to extract out the underlying column
    auto scalar_result =
      cudf::reduce(values,
                   cudf::make_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                   cudf::data_type{cudf::type_id::FLOAT64});
    auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
    std::vector<std::unique_ptr<cudf::column>> cols;
    std::transform(
      tbl.begin(), tbl.end(), std::back_inserter(cols), [](cudf::column_view const& col) {
        return std::make_unique<cudf::column>(col);
      });
    return cudf::make_structs_column(tbl.num_rows(), std::move(cols), 0, rmm::device_buffer());
  }
};

struct reduce_merge_op {
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& values, int delta) const
  {
    // result is a scalar, but we want to extract out the underlying column
    auto scalar_result =
      cudf::reduce(values,
                   cudf::make_merge_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                   cudf::data_type{cudf::type_id::FLOAT64});
    auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
    std::vector<std::unique_ptr<cudf::column>> cols;
    std::transform(
      tbl.begin(), tbl.end(), std::back_inserter(cols), [](cudf::column_view const& col) {
        return std::make_unique<cudf::column>(col);
      });
    return cudf::make_structs_column(tbl.num_rows(), std::move(cols), 0, rmm::device_buffer());
  }
};

TYPED_TEST(ReductionTDigestAllTypes, Simple)
{
  using T = TypeParam;
  tdigest_simple_aggregation<T>(reduce_op{});
}

TYPED_TEST(ReductionTDigestAllTypes, SimpleWithNulls)
{
  using T = TypeParam;
  tdigest_simple_with_nulls_aggregation<T>(reduce_op{});
}

TYPED_TEST(ReductionTDigestAllTypes, AllNull)
{
  using T = TypeParam;
  tdigest_simple_all_nulls_aggregation<T>(reduce_op{});
}

struct ReductionTDigestMerge : public cudf::test::BaseFixture {
};

TEST_F(ReductionTDigestMerge, Simple) { tdigest_merge_simple(reduce_op{}, reduce_merge_op{}); }

}  // namespace test
}  // namespace cudf