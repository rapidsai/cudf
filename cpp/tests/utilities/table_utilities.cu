#include "table_utilities.hpp"
#include "column_utilities.hpp"

#include <gmock/gmock.h>


namespace cudf {
namespace test {

void expect_table_properties_equal(cudf::table_view lhs, cudf::table_view rhs) {
  EXPECT_EQ(lhs.num_rows(), rhs.num_rows());
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
}

void expect_tables_equal(cudf::table_view lhs, cudf::table_view rhs) {
  expect_table_properties_equal(lhs, rhs);
  for (auto i=0; i<lhs.num_columns(); ++i) {
    cudf::test::expect_columns_equal(lhs.column(i), rhs.column(i));
  }
}

}  // namespace test
}  // namespace cudf