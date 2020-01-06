#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/round_robin.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column_factories.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>

#include <cassert>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <initializer_list>

#include <gtest/gtest.h>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

template <typename T>
class RoundRobinTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(RoundRobinTest, cudf::test::FixedWidthTypes);

TYPED_TEST(RoundRobinTest, RoundRobin2StringKeyNullColumns) {
  strings_column_wrapper rrColWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"}, {1,1,1,1,1,1,1,1,1,1,1,1,0});
  strings_column_wrapper rrColWrap3({"n", "o", "p", "q", "r", "s", "t", "u","v","w","x","y","z"}, {1,1,1,1,1,1,1,1,1,1,1,1,0});
  
  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rrColWrap3).size());

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
      if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
        return 1;
      else
        return row; });
    
  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2, rrColWrap3}};

  cudf::size_type num_partitions = 3;
  cudf::size_type start_partition = 0;

  std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<cudf::size_type>> result;
  EXPECT_NO_THROW(result = cudf::experimental::round_robin_partition(rr_view,
                                                                     num_partitions,
                                                                     start_partition));

  auto p_outputTable = std::move(result.first);

  cudf::column_view const& a_rr_tbl_cview{static_cast<cudf::column_view const&>(rrColWrap1)};
  
  const cudf::size_type outputRows = inputRows;
  strings_column_wrapper expectedDataWrap1({"ab", "ac", "bc", "bd", "cd", "ce", "de", "df", "ef", "eg", "fg", "fh", "gh", "gi", "hi", "hj"}, {1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0});  

  auto seq_out2 = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = (row % 2 == 0);
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row);
      });
  fixed_width_column_wrapper<TypeParam> expectedDataWrap2(seq_out2, seq_out2+outputRows);

  strings_column_wrapper expectedDataWrap3({"zy", "zx", "yx", "yw", "xw", "xv", "wv", "wu", "vu", "vt", "ut", "us", "ts", "tr", "sr", "sp"}, {1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0});

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
  auto expected_column_view3{static_cast<cudf::column_view const&>(expectedDataWrap3)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};
  auto output_column_view3{p_outputTable->view().column(2)};
  
  cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);
  cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
  cudf::test::expect_columns_equal(expected_column_view3, output_column_view3);
}

