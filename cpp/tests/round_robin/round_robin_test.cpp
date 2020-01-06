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
#include <numeric>

#include <gtest/gtest.h>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

template <typename T>
class RoundRobinTest : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(RoundRobinTest, cudf::test::FixedWidthTypes);

TYPED_TEST(RoundRobinTest, RoundRobin3Partitions) {
  strings_column_wrapper rrColWrap1({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"}, {1,1,1,1,1,1,1,1,1,1,1,1,0});
  //strings_column_wrapper rrColWrap1({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"}, {1,1,1,1,1,1,1,1,1,1,1,1,0});
  
  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::test::make_counting_transform_iterator(0, [](auto row) {
      if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8) {
        cudf::experimental::bool8 ret = (row % 2 == 0);
        return static_cast<TypeParam>(ret);
      }
      else
        return static_cast<TypeParam>(row); });
    
  fixed_width_column_wrapper<TypeParam> rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 3;
  std::vector<cudf::size_type> sequence(num_partitions,0);
  std::iota(sequence.begin(), sequence.end(), 0);
  
  std::vector<cudf::size_type> partition_offsets_0{0,5,9};
  EXPECT_EQ(num_partitions, partition_offsets_0.size());
  
  for(cudf::size_type start_partition = 0; start_partition < num_partitions; ++start_partition) {
    std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::experimental::round_robin_partition(rr_view,
                                                                       num_partitions,
                                                                       start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};
  
    strings_column_wrapper expectedDataWrap1({"a","d","g","j","m","b","e","h","k","c","f","i","l"},
                                             {1,1,1,1,0,1,1,1,1,1,1,1,1});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    cudf::test::expect_columns_equal(expected_column_view1, output_column_view1);

    if (cudf::experimental::type_to_id<TypeParam>() == cudf::BOOL8) {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({1,0,1,0,1,0,1,0,1,1,0,1,0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());
      
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam> expectedDataWrap2({0,3,6,9,12,1,4,7,10,2,5,8,11});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      cudf::test::expect_columns_equal(expected_column_view2, output_column_view2);
    }

    //rotate `partition_offsets_0` right by `start_partition` positions:
    //
    std::vector<cudf::size_type> expected_partition_offsets(num_partitions);
    std::transform(sequence.begin(), sequence.end(),
                   expected_partition_offsets.begin(),
                   [partition_offsets_0, start_partition](auto index) {
                     auto n = partition_offsets_0.size();
                     return partition_offsets_0[(index + n - start_partition) % n];
                   });

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

