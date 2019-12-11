#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/merge.hpp>
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
class MergeStringTest : public cudf::test::BaseFixture {};

///class MergeStringOnlyTest : public cudf::test::BaseFixture {}; // <- use with TEST_F

TYPED_TEST_CASE(MergeStringTest, cudf::test::FixedWidthTypes);

TYPED_TEST(MergeStringTest, Merge1StringKeyColumns) {
  strings_column_wrapper leftColWrap1({"ab", "bc", "cd", "de", "ef", "fg", "gh", "hi"});
  cudf::size_type inputRows1 = static_cast<cudf::column_view const&>(leftColWrap1).size();
    
  auto sequence0 = cudf::test::make_counting_transform_iterator(0, [](auto row) {
      if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
        return 0;
      else
        return row; });
        
  fixed_width_column_wrapper<TypeParam> leftColWrap2(sequence0, sequence0+inputRows1);


  
  strings_column_wrapper rightColWrap1({"ac", "bd", "ce", "df", "eg", "fh", "gi", "hj"});
  cudf::size_type inputRows2 = static_cast<cudf::column_view const&>(rightColWrap1).size();
  fixed_width_column_wrapper<TypeParam> rightColWrap2(sequence0, sequence0+inputRows2);
  
  cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};
    
  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order {cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  std::unique_ptr<cudf::experimental::table> p_outputTable;
  EXPECT_NO_THROW(p_outputTable = cudf::experimental::merge(left_view,
                                                            right_view,
                                                            key_cols,
                                                            column_order,
                                                            null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  const cudf::size_type outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();

  strings_column_wrapper expectedDataWrap1({"ab", "ac", "bc", "bd", "cd", "ce", "de", "df", "ef", "eg", "fg", "fh", "gh", "gi", "hi", "hj"});

  auto seq_out2 = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
      if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
        return 0;
      else
        return row / 2; });
  fixed_width_column_wrapper<TypeParam> expectedDataWrap2(seq_out2, seq_out2+outputRows);

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};    

  cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);
  cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
}

