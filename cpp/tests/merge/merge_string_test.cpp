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

//rename test <TestName> as DISABLED_<TestName> to disable:
//Example: TYPED_TEST(MergeStringTest, DISABLED_Merge2StringKeyColumns)
//
TYPED_TEST(MergeStringTest, Merge2StringKeyColumns) {
  strings_column_wrapper leftColWrap1({"ab", "bc", "cd", "de", "ef", "fg", "gh", "hi"});
  strings_column_wrapper leftColWrap3({"zy", "yx", "xw", "wv", "vu", "ut", "ts", "sr"});
  
  cudf::size_type inputRows = static_cast<cudf::column_view const&>(leftColWrap1).size();

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(leftColWrap3).size());

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
      if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
        return 1;
      else
        return 2 * row; });
    
  fixed_width_column_wrapper<TypeParam> leftColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view left_view{{leftColWrap1, leftColWrap2, leftColWrap3}};

  strings_column_wrapper rightColWrap1({"ac", "bd", "ce", "df", "eg", "fh", "gi", "hj"});

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rightColWrap1).size());

  auto sequence_r = cudf::test::make_counting_transform_iterator(0, [](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          return 0;
        else
          return 2 * row + 1; });
  fixed_width_column_wrapper<TypeParam> rightColWrap2(sequence_r, sequence_r + inputRows);

  strings_column_wrapper rightColWrap3({"zx", "yw", "xv", "wu", "vt", "us", "tr", "sp"});

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rightColWrap3).size());

  
  cudf::table_view right_view{{rightColWrap1, rightColWrap2, rightColWrap3}};

  std::vector<cudf::size_type> key_cols{0, 2};
  std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::DESCENDING};
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
          {
            cudf::experimental::bool8 ret = (row % 2 == 0);
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row);
      });
  fixed_width_column_wrapper<TypeParam> expectedDataWrap2(seq_out2, seq_out2+outputRows);

  strings_column_wrapper expectedDataWrap3({"zy", "zx", "yx", "yw", "xw", "xv", "wv", "wu", "vu", "vt", "ut", "us", "ts", "tr", "sr", "sp"});

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

TYPED_TEST(MergeStringTest, Merge1StringKeyNullColumns) {
  // data: "ab", "bc", "cd", "de" | valid: 1 1 1 0
  strings_column_wrapper leftColWrap1({"ab", "bc", "cd", "de", "ef", "fg", "gh", "hi"}, {1,1,1,1,1,1,1,0});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(leftColWrap1).size();

  auto sequence0 = cudf::test::make_counting_transform_iterator(0, [](auto row) {
      if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
        return 0;
      else
        return row; });
        
  fixed_width_column_wrapper<TypeParam> leftColWrap2(sequence0, sequence0+inputRows);
  cudf::table_view left_view{{leftColWrap1, leftColWrap2}};

  // data: "ac", "bd", "ce", "df" | valid: 1 1 1 0
  strings_column_wrapper rightColWrap1({"ac", "bd", "ce", "df", "eg", "fh", "gi", "hj"}, {1,1,1,1,1,1,1,0});
  fixed_width_column_wrapper<TypeParam> rightColWrap2(sequence0, sequence0+inputRows);

  cudf::table_view right_view{{rightColWrap1, rightColWrap2}};
  
  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order {cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

  std::unique_ptr<cudf::experimental::table> p_outputTable;
  EXPECT_NO_THROW( p_outputTable = cudf::experimental::merge(left_view,
                                                             right_view,
                                                             key_cols,
                                                             column_order,
                                                             null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  const cudf::size_type outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();
  const cudf::size_type column1TotalNulls = a_left_tbl_cview.null_count() + a_right_tbl_cview.null_count();

  // data: "ab", "ac", "bc", "bd", "cd", "ce", "de", "df" | valid: 1 1 1 1 1 1 0 0
  strings_column_wrapper expectedDataWrap1({"ab", "ac", "bc", "bd", "cd", "ce", "de", "df", "ef", "eg", "fg", "fh", "gh", "gi", "hi", "hj"},
                                           {1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0});
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

TYPED_TEST(MergeStringTest, Merge2StringKeyNullColumns) {
  strings_column_wrapper leftColWrap1({"ab", "bc", "cd", "de", "ef", "fg", "gh", "hi"}, {1,1,1,1,1,1,1,0});
  strings_column_wrapper leftColWrap3({"zy", "yx", "xw", "wv", "vu", "ut", "ts", "sr"}, {1,1,1,1,1,1,1,0});
  
  cudf::size_type inputRows = static_cast<cudf::column_view const&>(leftColWrap1).size();

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(leftColWrap3).size());

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
      if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
        return 1;
      else
        return 2 * row; });
    
  fixed_width_column_wrapper<TypeParam> leftColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view left_view{{leftColWrap1, leftColWrap2, leftColWrap3}};

  strings_column_wrapper rightColWrap1({"ac", "bd", "ce", "df", "eg", "fh", "gi", "hj"}, {1,1,1,1,1,1,1,0});

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rightColWrap1).size());

  auto sequence_r = cudf::test::make_counting_transform_iterator(0, [](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          return 0;
        else
          return 2 * row + 1; });
  fixed_width_column_wrapper<TypeParam> rightColWrap2(sequence_r, sequence_r + inputRows);

  strings_column_wrapper rightColWrap3({"zx", "yw", "xv", "wu", "vt", "us", "tr", "sp"}, {1,1,1,1,1,1,1,0});

  EXPECT_EQ(inputRows, static_cast<cudf::column_view const&>(rightColWrap3).size());

  
  cudf::table_view right_view{{rightColWrap1, rightColWrap2, rightColWrap3}};

  std::vector<cudf::size_type> key_cols{0, 2};
  std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER, cudf::null_order::BEFORE};

  std::unique_ptr<cudf::experimental::table> p_outputTable;
  EXPECT_NO_THROW(p_outputTable = cudf::experimental::merge(left_view,
                                                            right_view,
                                                            key_cols,
                                                            column_order,
                                                            null_precedence));

  cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
  cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
  const cudf::size_type outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();
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
