#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <chrono>
#include <cuda/std/ratio>
#include <vector>

#include <cudf/rolling/range_window_bounds.hpp>
#include <src/rolling/range_window_bounds_detail.hpp>

namespace cudf {
namespace test {

struct RangeWindowBoundsTest : public BaseFixture {
};

TEST_F(RangeWindowBoundsTest, TimestampsAndDurations)
{
  using namespace cudf;
  using namespace cudf::detail;

  {
    // Test that range_window_bounds specified in Days can be scaled down to seconds, milliseconds,
    // etc.
    auto range_3_days = range_bounds(duration_scalar<duration_D>{3, true});
    EXPECT_TRUE(range_3_days.range_scalar().is_valid());
    EXPECT_TRUE(!range_3_days.is_unbounded());

    range_3_days.scale_to(data_type{type_id::TIMESTAMP_SECONDS});
    EXPECT_EQ(range_comparable_value<int64_t>(range_3_days), 3 * 24 * 60 * 60);

    // Unchanged.
    range_3_days.scale_to(data_type{type_id::TIMESTAMP_SECONDS});
    EXPECT_EQ(range_comparable_value<int64_t>(range_3_days), 3 * 24 * 60 * 60);

    // Finer.
    range_3_days.scale_to(data_type{type_id::TIMESTAMP_MILLISECONDS});
    EXPECT_EQ(range_comparable_value<int64_t>(range_3_days), 3 * 24 * 60 * 60 * 1000);

    // Finer.
    range_3_days.scale_to(data_type{type_id::TIMESTAMP_MICROSECONDS});
    EXPECT_EQ(range_comparable_value<int64_t>(range_3_days),
              int64_t{3} * 24 * 60 * 60 * 1000 * 1000);

    // Scale back up to days. Should fail because of loss of precision.
    EXPECT_THROW(range_3_days.scale_to(data_type{type_id::TIMESTAMP_DAYS}), cudf::logic_error);
  }

  {
    // Negative tests.
    // Cannot scale from higher to lower precision. (Congruent with std::chrono::duration scaling.)
    // Cannot extract duration value in the wrong representation type.

    auto range_3M_ns = range_bounds(duration_scalar<duration_ns>{int64_t{3} * 1000 * 1000, true});
    EXPECT_THROW(range_3M_ns.scale_to(data_type{type_id::TIMESTAMP_DAYS}), cudf::logic_error);
    EXPECT_THROW(range_comparable_value<int32_t>(range_3M_ns), cudf::logic_error);

    auto range_3_days = range_bounds(duration_scalar<duration_D>{3, true});
    EXPECT_THROW(range_comparable_value<int64_t>(range_3_days), cudf::logic_error);
  }
}

template <typename T>
struct TypedRangeWindowBoundsTest : RangeWindowBoundsTest {
};

using TypesForTest = cudf::test::IntegralTypesNotBool;

TYPED_TEST_CASE(TypedRangeWindowBoundsTest, TypesForTest);

TYPED_TEST(TypedRangeWindowBoundsTest, BasicScaling)
{
  using namespace cudf;
  using namespace cudf::detail;

  using T = TypeParam;

  {
    auto numeric_bounds = range_bounds(numeric_scalar<T>{3, true});
    numeric_bounds.scale_to(data_type{type_to_id<T>()});
    EXPECT_EQ(range_comparable_value<T>(numeric_bounds), T{3});
  }

  {
    auto unbounded = range_window_bounds::unbounded(data_type{type_to_id<T>()});
    unbounded.scale_to(data_type{type_to_id<T>()});
    EXPECT_EQ(range_comparable_value<T>(unbounded), std::numeric_limits<T>::max());
  }

  {
    // Negative tests.
    auto numeric_bounds = range_bounds(numeric_scalar<T>{3, true});

    std::for_each(thrust::make_counting_iterator(1),
                  thrust::make_counting_iterator<int32_t>(static_cast<int>(type_id::NUM_TYPE_IDS)),
                  [&numeric_bounds](auto i) {
                    auto id = static_cast<type_id>(i);
                    if (type_to_id<T>() != id) {
                      EXPECT_THROW(numeric_bounds.scale_to(data_type{id}), cudf::logic_error);
                    }
                  });
  }
}

}  // namespace test
}  // namespace cudf
