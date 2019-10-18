
#include <cudf/copying.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>


template <typename T>
class GatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(GatherTest, cudf::test::NumericTypes);

TYPED_TEST(GatherTest, IdentityTest) {
  constexpr cudf::size_type source_size{1000};

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i){return i;});
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column{data, data+source_size};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map{data, data+source_size};

  std::vector<cudf::column_view> source_columns{source_column};
  cudf::table_view source_table {source_columns};

  std::unique_ptr<cudf::experimental::table> result =
    std::move(cudf::experimental::gather(
					 source_table,
					 gather_map
					 ));

  for (auto i=0; i<source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(source_table.column(i), result->view().column(i));
  }

}