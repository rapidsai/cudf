#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
class GatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(GatherTest, cudf::test::NumericTypes);

TYPED_TEST(GatherTest, IdentityTest) {
}