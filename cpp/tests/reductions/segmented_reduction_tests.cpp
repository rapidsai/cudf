#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>

#include <limits>

namespace cudf {
namespace test {

#define XXX 0  // null placeholder

template <typename T>
struct SegmentedReductionTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SegmentedReductionTest, NumericTypes);

TYPED_TEST(SegmentedReductionTest, Sum)
{
  // TODO: add nulls
  auto input   = fixed_width_column_wrapper<TypeParam>{3, 3, 3, 8, 8, 0, 0, 0, 0};
  auto offsets = fixed_width_column_wrapper<size_type>{0, 3, 5, 5, 9};
  auto expect  = fixed_width_column_wrapper<TypeParam>{{9, 16, XXX, 0}, {1, 1, 0, 1}};

  auto res =
    segmented_reduce(input, offsets, make_sum_aggregation(), data_type{type_to_id<TypeParam>()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, EmptySum)
{
  auto input   = fixed_width_column_wrapper<TypeParam>{};
  auto offsets = fixed_width_column_wrapper<size_type>{0};
  auto expect  = fixed_width_column_wrapper<TypeParam>{};

  auto res =
    segmented_reduce(input, offsets, make_sum_aggregation(), data_type{type_to_id<TypeParam>()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, Product)
{
  // TODO: add nulls
  auto input   = fixed_width_column_wrapper<TypeParam>{3, 3, 3, 8, 8, 0, 9, 9, 9};
  auto offsets = fixed_width_column_wrapper<size_type>{0, 3, 5, 5, 9};
  auto expect  = fixed_width_column_wrapper<TypeParam>{{27, 64, XXX, 0}, {1, 1, 0, 1}};

  auto res = segmented_reduce(
    input, offsets, make_product_aggregation(), data_type{type_to_id<TypeParam>()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, EmptyProduct)
{
  auto input   = fixed_width_column_wrapper<TypeParam>{};
  auto offsets = fixed_width_column_wrapper<size_type>{0};
  auto expect  = fixed_width_column_wrapper<TypeParam>{};

  auto res = segmented_reduce(
    input, offsets, make_product_aggregation(), data_type{type_to_id<TypeParam>()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, Max)
{
  // TODO: add nulls
  auto input   = fixed_width_column_wrapper<TypeParam>{1, 8, 6, 7, 8, 9, 3, 7, 6};
  auto offsets = fixed_width_column_wrapper<size_type>{0, 3, 3, 6, 9};
  auto expect  = fixed_width_column_wrapper<TypeParam>{{8, XXX, 9, 7}, {1, 0, 1, 1}};

  auto res =
    segmented_reduce(input, offsets, make_max_aggregation(), data_type{type_to_id<TypeParam>()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, Min)
{
  // TODO: add nulls
  auto input   = fixed_width_column_wrapper<TypeParam>{8, 8, 8, 3, 2, 4, 8, 6, 1, 7};
  auto offsets = fixed_width_column_wrapper<size_type>{0, 3, 3, 6, 10};
  auto expect  = fixed_width_column_wrapper<TypeParam>{{8, XXX, 2, 1}, {1, 0, 1, 1}};

  auto res =
    segmented_reduce(input, offsets, make_min_aggregation(), data_type{type_to_id<TypeParam>()});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, Any)
{
  // TODO: add nulls
  auto input   = fixed_width_column_wrapper<TypeParam>{4, 4, 4, 5, 0, 0, 0, 0, 0, 0};
  auto offsets = fixed_width_column_wrapper<size_type>{0, 3, 6, 6, 10};
  auto expect  = fixed_width_column_wrapper<bool>{{true, true, bool{XXX}, false}, {1, 1, 0, 1}};

  auto res = segmented_reduce(input, offsets, make_any_aggregation(), data_type{type_id::BOOL8});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, All)
{
  // TODO: add nulls
  auto input   = fixed_width_column_wrapper<TypeParam>{1, 2, 2, 0, 2, 6, 6, 6, 6, 6};
  auto offsets = fixed_width_column_wrapper<size_type>{0, 1, 5, 5, 10};
  auto expect  = fixed_width_column_wrapper<bool>{{true, false, bool{XXX}, true}, {1, 1, 0, 1}};

  auto res = segmented_reduce(input, offsets, make_all_aggregation(), data_type{type_id::BOOL8});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

#undef XXX

}  // namespace test
}  // namespace cudf