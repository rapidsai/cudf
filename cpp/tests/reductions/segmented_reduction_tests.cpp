#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>

namespace cudf {
namespace test {

template <typename T>
struct SegmentedReductionTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SegmentedReductionTest, NumericTypes);

TYPED_TEST(SegmentedReductionTest, Sum)
{
  // TODO: add nulls
  auto input   = fixed_width_column_wrapper<TypeParam>{3, 3, 3, 8, 8, -1, -1, -1, -1};
  auto offsets = fixed_width_column_wrapper<size_type>{0, 3, 5, 5, 10};
  auto expect  = fixed_width_column_wrapper<TypeParam>{9, 16, 0, -4};

  auto res =
    segmented_reduce(input, offsets, make_sum_aggregation(), data_type{type_to_id<TypeParam>()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

}  // namespace test
}  // namespace cudf