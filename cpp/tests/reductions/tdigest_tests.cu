/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/tdigest_utilities.cuh>
#include <cudf_test/type_lists.hpp>

#include <cudf/reduction.hpp>

template <typename T>
struct ReductionTDigestAllTypes : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(ReductionTDigestAllTypes, cudf::test::NumericTypes);

namespace {
struct reduce_op {
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& values, int delta) const
  {
    // result is a scalar, but we want to extract out the underlying column
    auto scalar_result =
      cudf::reduce(values,
                   *cudf::make_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                   cudf::data_type{cudf::type_id::STRUCT});
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
                   *cudf::make_merge_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                   cudf::data_type{cudf::type_id::STRUCT});
    auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
    std::vector<std::unique_ptr<cudf::column>> cols;
    std::transform(
      tbl.begin(), tbl.end(), std::back_inserter(cols), [](cudf::column_view const& col) {
        return std::make_unique<cudf::column>(col);
      });
    return cudf::make_structs_column(tbl.num_rows(), std::move(cols), 0, rmm::device_buffer());
  }
};
}  // namespace

TYPED_TEST(ReductionTDigestAllTypes, Simple)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_aggregation<T>(reduce_op{});
}

TYPED_TEST(ReductionTDigestAllTypes, SimpleWithNulls)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_with_nulls_aggregation<T>(reduce_op{});
}

TYPED_TEST(ReductionTDigestAllTypes, AllNull)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_all_nulls_aggregation<T>(reduce_op{});
}

struct ReductionTDigestMerge : public cudf::test::BaseFixture {};

TEST_F(ReductionTDigestMerge, Simple)
{
  cudf::test::tdigest_merge_simple(reduce_op{}, reduce_merge_op{});
}

// tests an issue with the cluster generating code with a small number of centroids that have large
// weights
TEST_F(ReductionTDigestMerge, FewHeavyCentroids)
{
  // digest 1
  cudf::test::fixed_width_column_wrapper<double> c0c{1.0, 2.0};
  cudf::test::fixed_width_column_wrapper<double> c0w{100.0, 50.0};
  cudf::test::structs_column_wrapper c0s({c0c, c0w});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c0_offsets{0, 2};
  auto c0l =
    cudf::make_lists_column(1, c0_offsets.release(), c0s.release(), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<double> c0min{1.0};
  cudf::test::fixed_width_column_wrapper<double> c0max{2.0};
  std::vector<std::unique_ptr<cudf::column>> c0_children;
  c0_children.push_back(std::move(c0l));
  c0_children.push_back(c0min.release());
  c0_children.push_back(c0max.release());
  // tdigest struct
  auto c0 = cudf::make_structs_column(1, std::move(c0_children), 0, {});
  cudf::tdigest::tdigest_column_view tdv0(*c0);

  // digest 2
  cudf::test::fixed_width_column_wrapper<double> c1c{3.0, 4.0};
  cudf::test::fixed_width_column_wrapper<double> c1w{200.0, 50.0};
  cudf::test::structs_column_wrapper c1s({c1c, c1w});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets{0, 2};
  auto c1l =
    cudf::make_lists_column(1, c1_offsets.release(), c1s.release(), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<double> c1min{3.0};
  cudf::test::fixed_width_column_wrapper<double> c1max{4.0};
  std::vector<std::unique_ptr<cudf::column>> c1_children;
  c1_children.push_back(std::move(c1l));
  c1_children.push_back(c1min.release());
  c1_children.push_back(c1max.release());
  // tdigest struct
  auto c1 = cudf::make_structs_column(1, std::move(c1_children), 0, {});

  std::vector<cudf::column_view> views;
  views.push_back(*c0);
  views.push_back(*c1);
  auto values = cudf::concatenate(views);

  // merge
  auto scalar_result =
    cudf::reduce(*values,
                 *cudf::make_merge_tdigest_aggregation<cudf::reduce_aggregation>(1000),
                 cudf::data_type{cudf::type_id::STRUCT});

  // convert to a table
  auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
  std::vector<std::unique_ptr<cudf::column>> cols;
  std::transform(
    tbl.begin(), tbl.end(), std::back_inserter(cols), [](cudf::column_view const& col) {
      return std::make_unique<cudf::column>(col);
    });
  auto result = cudf::make_structs_column(tbl.num_rows(), std::move(cols), 0, rmm::device_buffer());

  // we expect to see exactly 4 centroids (the same inputs) with properly computed min/max.
  cudf::test::fixed_width_column_wrapper<double> ec{1.0, 2.0, 3.0, 4.0};
  cudf::test::fixed_width_column_wrapper<double> ew{100.0, 50.0, 200.0, 50.0};
  cudf::test::structs_column_wrapper es({ec, ew});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets{0, 4};
  auto el = cudf::make_lists_column(1, e_offsets.release(), es.release(), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<double> emin{1.0};
  cudf::test::fixed_width_column_wrapper<double> emax{4.0};
  std::vector<std::unique_ptr<cudf::column>> e_children;
  e_children.push_back(std::move(el));
  e_children.push_back(emin.release());
  e_children.push_back(emax.release());
  // tdigest struct
  auto expected = cudf::make_structs_column(1, std::move(e_children), 0, {});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}
