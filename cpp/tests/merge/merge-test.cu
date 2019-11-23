//#include <nvstrings/NVCategory.h>
//#include <nvstrings/NVStrings.h>

#include <cudf/cudf.h>
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

#define DEBUG_

// for debugging, only; TODO: remove:
//
#ifdef DEBUG_
#include <iostream>
#include <iterator>
#endif

#ifdef DEBUG_
#endif

#include <cassert>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <initializer_list>

#include <gtest/gtest.h>


#ifdef DEBUG_
namespace{ //anonym.
template<typename ColType>
using hostColType = std::pair<std::vector<ColType>, std::vector<cudf::bitmask_type>>;
  
template<typename T, typename...Args, template<typename,typename...> class Vector>
void print_v(const Vector<T, Args...>& v, std::ostream& os)
{
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}

template<typename T, typename...Args, template<typename,typename...> class Vector>
void print_v(const Vector<T, Args...>& v, typename Vector<T, Args...>::const_iterator pos, std::ostream& os)
{ 
  std::copy(v.begin(), pos, std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}

template<typename T, typename...Args, template<typename,typename...> class Vector>
void print_v(const Vector<T, Args...>& v, size_t n, std::ostream& os)
{ 
  std::copy_n(v.begin(), n, std::ostream_iterator<T>(os,","));//okay
  os<<"\n";
}


 template<typename ColType>
 void print_col(cudf::column_view c, std::ostream& os)
 {
   hostColType<ColType> col_host_pair = cudf::test::to_host<ColType>(c);

   std::cout<<"column data:\n";
   if( col_host_pair.first.empty() )
     std::cout<<"empty data...\n";
   else
     print_v(col_host_pair.first, std::cout);

   if( col_host_pair.second.empty() )
     std::cout<<"empty null mask...\n";
   else
     {
       std::cout<<"column null mask:\n";
       print_v(col_host_pair.second, std::cout);
     }
 }
}//anonym.
#endif

template <typename T>
class MergeTest_ : public cudf::test::BaseFixture {};

//TODO: confirm if the legacy test_types below can be replaced
//just by cudf::test::NumericTypes
//
//legacy:
//{
///using test_types =
///  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>
//,cudf::bool8>; //column_wrapper failure //, cudf::nvstring_category>; //string not ready

//TYPED_TEST_CASE(MergeTest_, test_types);
//}

//TYPED_TEST_CASE(MergeTest_, cudf::test::NumericTypes); // <- TODO: put me back!
TYPED_TEST_CASE(MergeTest_, cudf::test::Types<int32_t>); // for now debug one type at a time...

TYPED_TEST(MergeTest_, MismatchedNumColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};
    columnFactoryT rightColWrap2{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}



TYPED_TEST(MergeTest_, MismatchedColumnDypes) {
    cudf::test::fixed_width_column_wrapper<int32_t> leftColWrap1{{0,1,2,3}};
    cudf::test::fixed_width_column_wrapper<double> rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};


    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}


TYPED_TEST(MergeTest_, EmptyKeyColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{}; // empty! this should trigger exception
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}


TYPED_TEST(MergeTest_, TooManyKeyColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0, 1}; // more keys than columns: this should trigger exception
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}


TYPED_TEST(MergeTest_, EmptyOrderTypes) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {}; // empty! this should trigger exception
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}


TYPED_TEST(MergeTest_, TooManyOrderTypes) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}}; 
    columnFactoryT rightColWrap1{{0,1,2,3}};

    std::vector<cudf::size_type> key_cols{0}; 
    std::vector<cudf::order> column_order {cudf::order::ASCENDING, cudf::order::DESCENDING}; // more order types than columns: this should trigger exception
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}

TYPED_TEST(MergeTest_, MismatchedKeyColumnsAndOrderTypes) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    columnFactoryT leftColWrap1{{0,1,2,3}};
    columnFactoryT leftColWrap2{{0,1,2,3}};
    columnFactoryT rightColWrap1{{0,1,2,3}};
    columnFactoryT rightColWrap2{{0,1,2,3}};

    cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
    cudf::table_view right_view{{rightColWrap1, rightColWrap2}};
    
    std::vector<cudf::size_type> key_cols{0, 1};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    std::vector<cudf::size_type> sortByCols = {0, 1};
    std::vector<order_by_type> orderByTypes = {GDF_ORDER_ASC};

    EXPECT_THROW(cudf::experimental::merge(left_view,
                                           right_view,
                                           key_cols,
                                           column_order,
                                           null_precedence), cudf::logic_error);
}

TYPED_TEST(MergeTest_, MergeWithEmptyColumn) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;

    cudf::size_type inputRows = 50000;
    auto unwrap_max = cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()); 
    inputRows = (static_cast<cudf::size_type>(unwrap_max) < inputRows ? 40 : inputRows);

    auto sequence = cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
    columnFactoryT leftColWrap1(sequence, sequence+inputRows);
    columnFactoryT rightColWrap1{};//wrapper of empty column <- this might require a (sequence, sequence) generator 

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    std::unique_ptr<cudf::experimental::table> p_outputTable;
    EXPECT_NO_THROW(p_outputTable = cudf::experimental::merge(left_view,
                                                              right_view,
                                                              key_cols,
                                                              column_order,
                                                              null_precedence));

    cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
    cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
    const cudf::size_type outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();
    
    columnFactoryT expectedDataWrap1(sequence, sequence+outputRows);//<- confirmed I can reuse a sequence, wo/ creating overlapping columns!

    auto expected_column_view{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    auto output_column_view{p_outputTable->view().column(0)};
    
    cudf::test::expect_columns_equal(expected_column_view, output_column_view);
}

TYPED_TEST(MergeTest_, Merge1KeyColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    cudf::size_type inputRows = 50000;
    auto unwrap_max = cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()); 
    inputRows = (static_cast<cudf::size_type>(unwrap_max) < inputRows ? 40 : inputRows);

#ifdef DEBUG_
    inputRows = 8;//simplify debugging...
#endif
    
    auto sequence0 = cudf::test::make_counting_transform_iterator(0, [](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          return 0;
        else
          return row; });
        
    auto sequence1 = cudf::test::make_counting_transform_iterator(0, [](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          return 1;
        else
          return 2 * row; });
    
    auto sequence2 = cudf::test::make_counting_transform_iterator(0, [](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          return 0;
        else
          return 2 * row + 1; });

    columnFactoryT leftColWrap1(sequence1, sequence1+inputRows);
    columnFactoryT leftColWrap2(sequence0, sequence0+inputRows);

  
    columnFactoryT rightColWrap1(sequence2, sequence2+inputRows);
    columnFactoryT rightColWrap2(sequence0, sequence0+inputRows);//<- confirmed I can reuse a sequence, wo/ creating overlapping columns!
  
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

#ifdef DEBUG_
    std::cout<<"##### inputRows: "<<inputRows<<"\n";
    
    std::cout<<"left table columns:\n";
    std::cout<<"1:\n";
    print_col<TypeParam>(a_left_tbl_cview, std::cout);

    std::cout<<"2:\n";
    print_col<TypeParam>(static_cast<cudf::column_view const&>(leftColWrap2), std::cout);

    std::cout<<"right table columns:\n";
    std::cout<<"1:\n";
    print_col<TypeParam>(a_right_tbl_cview, std::cout);

    std::cout<<"2:\n";
    print_col<TypeParam>(static_cast<cudf::column_view const&>(rightColWrap2), std::cout);
#endif
    
    
    auto seq_out1 = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = (row >= outputRows / 2); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row);
      });
    columnFactoryT expectedDataWrap1(seq_out1, seq_out1+outputRows);

    auto seq_out2 = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          return 0;
        else
          return row / 2; });
    columnFactoryT expectedDataWrap2(seq_out2, seq_out2+outputRows);

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

#ifdef DEBUG_
    std::cout<<"##### outputRows: "<<outputRows<<"\n";
    std::cout<<"##### output views sizes: "
             <<output_column_view1.size()
             <<", "
             << output_column_view2.size()
             <<"\n";

    std::cout<<"output table columns:\n";
    std::cout<<"1:\n";
    print_col<TypeParam>(output_column_view1, std::cout);

    std::cout<<"2:\n";
    print_col<TypeParam>(output_column_view2, std::cout);
#endif
    

    //PROBLEM: columns don't get lex-sorted!
    //
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);
    cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
}

TYPED_TEST(MergeTest_, Merge2KeyColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    cudf::size_type inputRows = 50000;
    auto unwrap_max = cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()); 
    inputRows = (static_cast<cudf::size_type>(unwrap_max) < inputRows ? 40 : inputRows);

    auto sequence1 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = (row >= inputRows / 2); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row);
      });
    columnFactoryT leftColWrap1(sequence1, sequence1 + inputRows);

    auto sequence2 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = ((row / (inputRows / 4)) % 2 == 0); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(2 * row);
      });
    columnFactoryT leftColWrap2(sequence2, sequence2 + inputRows);

    columnFactoryT rightColWrap1(sequence1, sequence1 + inputRows);

    auto sequence3 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = ((row / (inputRows / 4)) % 2 == 0); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(2 * row + 1);
      });
    columnFactoryT rightColWrap2(sequence3, sequence3 + inputRows);

    cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
    cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

    std::vector<cudf::size_type> key_cols{0, 1};
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
    
    auto seq_out1 = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = (row >= outputRows / 2); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row / 2);
      });
    columnFactoryT expectedDataWrap1(seq_out1, seq_out1+outputRows);

    auto seq_out2 = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = ((row / (outputRows / 4)) % 2 == 0); 
            return static_cast<TypeParam>(ret);
          }
        else
          {
            auto ret = (row % 2 == 0 ? row + 1 : row - 1);
            return static_cast<TypeParam>(ret);
          }
      });
    columnFactoryT expectedDataWrap2(seq_out2, seq_out2+outputRows);

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);
    cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
}

TYPED_TEST(MergeTest_, Merge1KeyNullColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    cudf::size_type inputRows = 50000;
    auto unwrap_max = cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()); 
    inputRows = (static_cast<cudf::size_type>(unwrap_max) < inputRows ? 40 : inputRows);

    // data: 0  2  4  6 | valid: 1 1 1 0
    auto sequence1 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = 0; 
            return static_cast<TypeParam>(ret); // <- no shortcut to this can avoid compiler errors
          }
        else
          return static_cast<TypeParam>(2 * row);
      });
    auto valid_sequence1 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        return (row < inputRows - 1);
      });
    columnFactoryT leftColWrap1(sequence1, sequence1 + inputRows, valid_sequence1);

    // data: 1  3  5  7 | valid: 1 1 1 0
    auto sequence2 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = 1; 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(2 * row + 1);
      });
    columnFactoryT rightColWrap1(sequence2, sequence2 + inputRows, valid_sequence1); // <- recycle valid_seq1, confirmed okay...

    std::vector<cudf::size_type> key_cols{0};
    std::vector<cudf::order> column_order {cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{};

    cudf::table_view left_view{{leftColWrap1}};
    cudf::table_view right_view{{rightColWrap1}};

    std::unique_ptr<cudf::experimental::table> p_outputTable;
    EXPECT_NO_THROW(p_outputTable = cudf::experimental::merge(left_view,
                                                              right_view,
                                                              key_cols,
                                                              column_order,
                                                              null_precedence));

    cudf::column_view const& a_left_tbl_cview{static_cast<cudf::column_view const&>(leftColWrap1)};
    cudf::column_view const& a_right_tbl_cview{static_cast<cudf::column_view const&>(rightColWrap1)};
    const cudf::size_type outputRows = a_left_tbl_cview.size() + a_right_tbl_cview.size();
    const cudf::size_type column1TotalNulls = a_left_tbl_cview.null_count() + a_right_tbl_cview.null_count();

    // data: 0 1 2 3 4 5 6 7 | valid: 1 1 1 1 1 1 0 0
    auto seq_out1 = cudf::test::make_counting_transform_iterator(0, [outputRows, column1TotalNulls](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = (row >= (outputRows - column1TotalNulls) / 2); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row);
      });
    auto valid_seq_out = cudf::test::make_counting_transform_iterator(0, [outputRows, column1TotalNulls](auto row) {
        return (row < (outputRows - column1TotalNulls));
      });
    columnFactoryT expectedDataWrap1(seq_out1, seq_out1 + outputRows, valid_seq_out);
    
    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    auto output_column_view1{p_outputTable->view().column(0)};
    
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);
}

TYPED_TEST(MergeTest_, Merge2KeyNullColumns) {
    using columnFactoryT = cudf::test::fixed_width_column_wrapper<TypeParam>;
    
    cudf::size_type inputRows = 50000;
    auto unwrap_max = cudf::detail::unwrap(std::numeric_limits<TypeParam>::max()); 
    inputRows = (static_cast<cudf::size_type>(unwrap_max) < inputRows ? 40 : inputRows);

    // data: 0 1 2 3 | valid: 1 1 1 1
    auto sequence1 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = (row >= inputRows / 2); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row);
      });
    auto valid_sequence1 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        return true;
      });
    columnFactoryT leftColWrap1(sequence1, sequence1 + inputRows);// <- purposelly left out: valid_sequence1;

    // data: 0 2 4 6 | valid: 1 1 1 1
    auto sequence2 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = ((row / (inputRows / 4)) % 2 == 0); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(2 * row);
      });
    columnFactoryT leftColWrap2(sequence2, sequence2 + inputRows, valid_sequence1);


    // data: 0 1 2 3 | valid: 1 1 1 1
    columnFactoryT rightColWrap1(sequence1, sequence1 + inputRows);// <- purposelly left out: valid_sequence1;
    
    // data: 0 1 2 3 | valid: 0 0 0 0
    auto sequence3 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = ((row / (inputRows / 4)) % 2 == 0); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row);
      });
    auto valid_sequence0 = cudf::test::make_counting_transform_iterator(0, [inputRows](auto row) {
        return false;
      });
    columnFactoryT rightColWrap2(sequence3, sequence3 + inputRows, valid_sequence0);

    cudf::table_view left_view{{leftColWrap1, leftColWrap2}};
    cudf::table_view right_view{{rightColWrap1, rightColWrap2}};

    std::vector<cudf::size_type> key_cols{0, 1};
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
    
    // data: 0 0 1 1 2 2 3 3 | valid: 1 1 1 1 1 1 1 1
    auto seq_out1 = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = (row >= outputRows / 2); 
            return static_cast<TypeParam>(ret);
          }
        else
          return static_cast<TypeParam>(row / 2);
      });
    columnFactoryT expectedDataWrap1(seq_out1, seq_out1+outputRows, valid_sequence1);
    
    // data: 0 0 2 1 4 2 6 3 | valid: 0 1 0 1 0 1 0 1
    auto seq_out2 = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          {
            cudf::experimental::bool8 ret = ((row / (outputRows / 8)) % 2 == 0); 
            return static_cast<TypeParam>(ret);
          }
        else
          {
            auto ret = (row % 2 != 0 ? 2 * (row / 2) : (row / 2));
            return static_cast<TypeParam>(ret);
          }
      });
    auto valid_sequence_out = cudf::test::make_counting_transform_iterator(0, [outputRows](auto row) {
        if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8)
          { 
            return ((row / (outputRows / 4)) % 2 == 1);
          }
        else
          {
            return (row % 2 != 0);
          }
      });
    columnFactoryT expectedDataWrap2(seq_out2, seq_out2 + outputRows, valid_sequence_out);

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);
    cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
}

