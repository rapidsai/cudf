#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <limits>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <../../src/groupby/hash/aggregation_operations.cuh>

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  This file is for unit testing the top level libgdf hash based groupby API
 */
/* ----------------------------------------------------------------------------*/

//gdf_error gdf_group_by_sum(int ncols,                    // # columns
//                           gdf_column** cols,            //input cols
//                           gdf_column* col_agg,          //column to aggregate on
//                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
//                           gdf_column** out_col_values,  //if not null return the grouped-by columns
//                                                         //(multi-gather based on indices, which are needed anyway)
//                           gdf_column* out_col_agg,      //aggregation result
//                           gdf_context* ctxt);           //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
//
//gdf_error gdf_group_by_min(int ncols,                    // # columns
//                           gdf_column** cols,            //input cols
//                           gdf_column* col_agg,          //column to aggregate on
//                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
//                           gdf_column** out_col_values,  //if not null return the grouped-by columns
//                                                         //(multi-gather based on indices, which are needed anyway)
//                           gdf_column* out_col_agg,      //aggregation result
//                           gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
//
//
//gdf_error gdf_group_by_max(int ncols,                    // # columns
//                           gdf_column** cols,            //input cols
//                           gdf_column* col_agg,          //column to aggregate on
//                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
//                           gdf_column** out_col_values,  //if not null return the grouped-by columns
//                                                         //(multi-gather based on indices, which are needed anyway)
//                           gdf_column* out_col_agg,      //aggregation result
//                           gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
//
//
//gdf_error gdf_group_by_avg(int ncols,                    // # columns
//                           gdf_column** cols,            //input cols
//                           gdf_column* col_agg,          //column to aggregate on
//                           gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
//                           gdf_column** out_col_values,  //if not null return the grouped-by columns
//                                                         //(multi-gather based on indices, which are needed anyway)
//                           gdf_column* out_col_agg,      //aggregation result
//                           gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct
//
//gdf_error gdf_group_by_count(int ncols,                    // # columns
//                             gdf_column** cols,            //input cols
//                             gdf_column* col_agg,          //column to aggregate on
//                             gdf_column* out_col_indices,  //if not null return indices of re-ordered rows
//                             gdf_column** out_col_values,  //if not null return the grouped-by columns
//                                                         //(multi-gather based on indices, which are needed anyway)
//                             gdf_column* out_col_agg,      //aggregation result
//                             gdf_context* ctxt);            //struct with additional info: bool is_sorted, flag_sort_or_hash, bool flag_count_distinct

enum struct agg_op
{
  MIN,
  MAX,
  SUM,
  COUNT,
  AVG
};

// unique_ptr wrappers for gdf_columns that will free their data
std::function<void(gdf_column*)> gdf_col_deleter = [](gdf_column* col){col->size = 0; cudaFree(col->data);};
using gdf_col_pointer = typename std::unique_ptr<gdf_column, decltype(gdf_col_deleter)>;

// Creates a gdf_column from a std::vector
template <typename col_type>
gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector)
{

  // Create a new instance of a gdf_column with a custom deleter that will free
  // the associated device memory when it eventually goes out of scope
  gdf_col_pointer the_column{new gdf_column, gdf_col_deleter};
  // Allocate device storage for gdf_column and copy contents from host_vector
  cudaMalloc(&(the_column->data), host_vector.size() * sizeof(col_type));
  cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice);

  // Deduce the type and set the gdf_dtype accordingly
  gdf_dtype gdf_col_type;
  if(std::is_same<col_type,int8_t>::value) gdf_col_type = GDF_INT8;
  else if(std::is_same<col_type,uint8_t>::value) gdf_col_type = GDF_INT8;
  else if(std::is_same<col_type,int16_t>::value) gdf_col_type = GDF_INT16;
  else if(std::is_same<col_type,uint16_t>::value) gdf_col_type = GDF_INT16;
  else if(std::is_same<col_type,int32_t>::value) gdf_col_type = GDF_INT32;
  else if(std::is_same<col_type,uint32_t>::value) gdf_col_type = GDF_INT32;
  else if(std::is_same<col_type,int64_t>::value) gdf_col_type = GDF_INT64;
  else if(std::is_same<col_type,uint64_t>::value) gdf_col_type = GDF_INT64;
  else if(std::is_same<col_type,float>::value) gdf_col_type = GDF_FLOAT32;
  else if(std::is_same<col_type,double>::value) gdf_col_type = GDF_FLOAT64;
  // Fill the gdf_column members
  the_column->valid = nullptr;
  the_column->size = host_vector.size();
  the_column->dtype = gdf_col_type;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  the_column->dtype_info = extra_info;
  return the_column;
}

// A new instance of this class will be created for each *TEST(GroupByTest, ...)
// Put all repeated stuff for each test here
template <typename test_parameters>
struct GDFGroupByTest : public testing::Test 
{
  using key_type = typename test_parameters::key_type;
  using value_type = typename test_parameters::value_type;

  const agg_op aggregation_operation{test_parameters::the_aggregator};

  const key_type unused_key{std::numeric_limits<key_type>::max()};
  const value_type unused_value{std::numeric_limits<value_type>::max()};


  // Result columns from gdf groupby
  gdf_col_pointer groupby_result{new gdf_column, gdf_col_deleter};
  gdf_col_pointer aggregation_result{new gdf_column, gdf_col_deleter};

  GDFGroupByTest() 
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }

  ~GDFGroupByTest() {} 

  std::pair<std::vector<key_type>, std::vector<value_type>>
    create_reference_input(const size_t num_keys, const size_t num_values_per_key, const int max_key = RAND_MAX, const int max_value = RAND_MAX, bool print = false) 
    {
      const size_t input_size = num_keys * num_values_per_key;

      std::vector<key_type> groupby_column;
      std::vector<value_type> aggregation_column;

      groupby_column.reserve(input_size);
      aggregation_column.reserve(input_size);

      for(int i = 0; i < num_keys; ++i )
      {
        // Create random key
        key_type current_key = std::rand() % max_key;

        // Don't use unused_key
        while(current_key == this->unused_key)
        {
          current_key = std::rand();
        }

        // For the current key, generate random values
        for(int j = 0; j < num_values_per_key; ++j)
        {
          value_type current_value = std::rand() % max_value;

          // Don't use unused_value
          while(current_value == this->unused_value)
          {
            current_value = std::rand() % max_value;
          }

          // Store current key and value
          groupby_column.push_back(current_key);
          aggregation_column.push_back(current_value);
        }
      }

      if(print)
      {
        std::cout << "Number of unique keys: " << num_keys 
          << " Values per key: " << num_values_per_key << "\n";

        std::cout << "Group By Column. Size: " << groupby_column.size() << " \n";
        std::copy(groupby_column.begin(), groupby_column.end(), std::ostream_iterator<key_type>(std::cout, " "));
        std::cout << "\n";

        std::cout << "Aggregation Column. Size: " << aggregation_column.size() << "\n";
        std::copy(aggregation_column.begin(), aggregation_column.end(), std::ostream_iterator<value_type>(std::cout, " "));
        std::cout << "\n";
      }

      return std::make_pair(groupby_column, aggregation_column);
    }


  template <class aggregation_operation>
    std::map<key_type, value_type> 
    compute_reference_solution(std::vector<key_type> const & groupby_column,
        std::vector<value_type> const & aggregation_column,
        bool print = false)
    {
      std::map<key_type, value_type> expected_values;

      // Computing the reference solution for AVG has to be handled uniquely
      if(std::is_same<aggregation_operation, avg_op<value_type>>::value)
      {

        // For each unique key, compute the SUM and COUNT aggregation
        std::map<key_type, value_type> counts = compute_reference_solution<count_op<value_type>>(groupby_column, aggregation_column);
        std::map<key_type, value_type> sums = compute_reference_solution<sum_op<value_type>>(groupby_column, aggregation_column);

        // For each unique key, compute it's AVG as SUM / COUNT
        for(auto & sum: sums)
        {
          const auto current_key = sum.first;

          auto count = counts.find(current_key);

          EXPECT_NE(count, counts.end()) << "Failed to find match for key " << current_key << " from the SUM solution in the COUNT solution";

          // Compute the AVG in place on the SUM map
          sum.second = sum.second / count->second;
        }

        expected_values = sums;
      }
      else
      {
        aggregation_operation op;

        for(size_t i = 0; i < groupby_column.size(); ++i){

          key_type current_key = groupby_column[i];
          value_type current_value = aggregation_column[i];

          // Use a STL map to keep track of the aggregation for each key
          auto found = expected_values.find(current_key);

          // Key doesn't exist yet, insert it
          if(found == expected_values.end())
          {
            // To support operations like `count`, on the first insert, perform the
            // operation on the new value and the operation's identity value and store the result
            current_value = op(current_value, aggregation_operation::IDENTITY);

            expected_values.insert(std::make_pair(current_key,current_value)); 

            if(print)
              std::cout << "First Insert of Key: " << current_key << " value: " << current_value << std::endl;
          }
          // Key exists, update the value with the operator
          else
          {
            value_type new_value = op(current_value, found->second);
            if(print)
              std::cout << "Insert of Key: " << current_key << " inserting value: " << current_value 
                << " storing: " << new_value << std::endl;
            found->second = new_value;
          }
        }
      }

      if(print)
      {
        for(auto const & a : expected_values)
        {
          std::cout << a.first << ", " << a.second << std::endl;
        }
      }


      return expected_values;
    }

  // Dispatches computing the reference solution based on which aggregation is to be performed
  // determined by the test_parameters class template argument
  std::map<key_type, value_type> 
    compute_reference_solution(std::vector<key_type> const & groupby_column, 
        std::vector<value_type> const & aggregation_column,
        bool print = false)
    {
      switch(test_parameters::the_aggregator)
      {
        // FIXME May need to use this->template compute_reference_solution<...>(...);
        case agg_op::MIN:  return compute_reference_solution<min_op<value_type>>(groupby_column,aggregation_column, print);
        case agg_op::MAX:  return compute_reference_solution<max_op<value_type>>(groupby_column,aggregation_column, print);
        case agg_op::SUM:  return compute_reference_solution<sum_op<value_type>>(groupby_column,aggregation_column, print);
        case agg_op::COUNT:return compute_reference_solution<count_op<value_type>>(groupby_column,aggregation_column, print);
        case agg_op::AVG:  return compute_reference_solution<avg_op<value_type>>(groupby_column,aggregation_column, print);
        default: std::cout << "Invalid aggregation operation.\n";
      }

      return std::map<key_type, value_type>();
    }

  void compute_gdf_result(gdf_column * groupby_column, 
      gdf_column * aggregation_column, 
      bool print = false, 
      bool sort = true)
  {

  }

};



// Google Test can only do a parameterized typed-test over a single type, so we have
// to nest multiple types inside of the TestParameters struct
template <typename groupby_type, typename aggregation_type, agg_op the_agg>
struct TestParameters
{
  using key_type = groupby_type;
  using value_type = aggregation_type;
  const static agg_op the_aggregator{the_agg};
};
using TestCases = ::testing::Types< TestParameters<int32_t, int32_t, agg_op::MAX>,
                                    //TestParameters<int32_t, float, agg_op::MAX>,
                                    //TestParameters<int32_t, double, agg_op::MAX>,
                                    //TestParameters<int32_t, int64_t, agg_op::MAX>,
                                    //TestParameters<int32_t, uint64_t, agg_op::MAX>,
                                    //TestParameters<int64_t, int32_t, agg_op::MAX>,
                                    //TestParameters<int64_t, float, agg_op::MAX>,
                                    //TestParameters<int64_t, double, agg_op::MAX>,
                                    //TestParameters<int64_t, int64_t, agg_op::MAX>,
                                    //TestParameters<int64_t, uint64_t, agg_op::MAX>,
                                    //TestParameters<uint64_t, int32_t, agg_op::MAX>,
                                    //TestParameters<uint64_t, float, agg_op::MAX>,
                                    //TestParameters<uint64_t, double, agg_op::MAX>,
                                    //TestParameters<uint64_t, int64_t, agg_op::MAX>,
                                    //TestParameters<uint64_t, uint64_t, agg_op::MAX>,
                                    TestParameters<int32_t, int32_t, agg_op::MIN>,
                                    //TestParameters<int32_t, float, agg_op::MIN>,
                                    //TestParameters<int32_t, double, agg_op::MIN>,
                                    //TestParameters<int32_t, int64_t, agg_op::MIN>,
                                    //TestParameters<int32_t, uint64_t, agg_op::MIN>,
                                    //TestParameters<uint64_t, int32_t, agg_op::MIN>,
                                    //TestParameters<uint64_t, float, agg_op::MIN>,
                                    //TestParameters<uint64_t, double, agg_op::MIN>,
                                    //TestParameters<uint64_t, int64_t, agg_op::MIN>,
                                    //TestParameters<uint64_t, uint64_t, agg_op::MIN>,
                                    TestParameters<int32_t, int32_t, agg_op::COUNT>,
                                    //TestParameters<int32_t, float, agg_op::COUNT>,
                                    //TestParameters<int32_t, double, agg_op::COUNT>,
                                    //TestParameters<int32_t, int64_t, agg_op::COUNT>,
                                    //TestParameters<int32_t, uint64_t, agg_op::COUNT>,
                                    //TestParameters<uint64_t, int32_t, agg_op::COUNT>,
                                    //TestParameters<uint64_t, float, agg_op::COUNT>,
                                    //TestParameters<uint64_t, double, agg_op::COUNT>,
                                    //TestParameters<uint64_t, int64_t, agg_op::COUNT>,
                                    //TestParameters<uint64_t, uint64_t, agg_op::COUNT>,
                                    TestParameters<int32_t, int32_t, agg_op::SUM>,
                                    //// TODO: Tests for SUM on single precision floats currently fail due to numerical stability issues
                                    ////TestParameters<int32_t, float, agg_op::SUM>, 
                                    //TestParameters<int32_t, double, agg_op::SUM>,
                                    //TestParameters<int32_t, int64_t, agg_op::SUM>,
                                    //TestParameters<int32_t, uint64_t, agg_op::SUM>,
                                    //TestParameters<uint64_t, double, agg_op::SUM>,
                                    //TestParameters<uint64_t, double, agg_op::SUM>,
                                    //TestParameters<uint64_t, int64_t, agg_op::SUM>,
                                    //TestParameters<uint64_t, uint64_t, agg_op::SUM>
                                    TestParameters<int32_t, float, agg_op::AVG>
                                    >;

  TYPED_TEST_CASE(GDFGroupByTest, TestCases);

TYPED_TEST(GDFGroupByTest, ExampleTest)
{
  const int num_keys = 10;
  const int num_values_per_key = 1;
  auto input = this->create_reference_input(num_keys, num_values_per_key, 5, 5, true);
  auto expected_values = this->compute_reference_solution(input.first, input.second, true);

  gdf_col_pointer gdf_groupby_column = create_gdf_column(input.first);
  gdf_col_pointer gdf_aggregation_column = create_gdf_column(input.second);


//
//  this->build_aggregation_table_device(input);
//  this->verify_aggregation_table(expected_values);
//
//  size_t computed_result_size = this->extract_groupby_result_device();
//  this->verify_groupby_result(computed_result_size,expected_values);
}



//TEST(HashGroupByTest, max)
//{
//
//  std::vector<int64_t> groupby_column{ 2, 2, 1, 1, 4, 3, 3, 42, 42  };
//  std::vector<double>  aggregation_column{5., 2., 2., 3., 7., 6., 6., 42., 51. };
//
//  const size_t size = groupby_column.size();
//
//  thrust::device_vector<int64_t> d_groupby_column(groupby_column);
//  thrust::device_vector<double> d_aggregation_column(aggregation_column);
//
//  gdf_column gdf_groupby_column;
//  gdf_groupby_column.data = static_cast<void*>(d_groupby_column.data().get());
//  gdf_groupby_column.size = size;
//  gdf_groupby_column.dtype = GDF_INT64;
//
//  gdf_column gdf_aggregation_column;
//  gdf_aggregation_column.data = static_cast<void*>(d_aggregation_column.data().get());
//  gdf_aggregation_column.size = size;
//  gdf_aggregation_column.dtype = GDF_FLOAT64;
//
//  thrust::device_vector<int64_t> groupby_result{size};
//  thrust::device_vector<double> aggregation_result{size};
//
//  gdf_column gdf_groupby_result;
//  gdf_groupby_result.data = static_cast<void*>(groupby_result.data().get());
//  gdf_groupby_result.size = size;
//  gdf_groupby_result.dtype = GDF_INT64;
//
//  gdf_column gdf_aggregation_result;
//  gdf_aggregation_result.data = static_cast<void*>(aggregation_result.data().get());
//  gdf_aggregation_result.size = size;
//  gdf_aggregation_result.dtype = GDF_FLOAT64;
//
//  // Determines if the final result is sorted
//  int flag_sort_result = 1;
//
//  gdf_context context{0, GDF_HASH, 0, flag_sort_result};
//
//  gdf_column * p_gdf_groupby_column = &gdf_groupby_column;
//
//  gdf_column * p_gdf_groupby_result = &gdf_groupby_result;
//
//  gdf_group_by_max((int) 1,      
//                   &p_gdf_groupby_column,
//                   &gdf_aggregation_column,
//                   nullptr,         
//                   &p_gdf_groupby_result,
//                   &gdf_aggregation_result,
//                   &context);
//
//  print_v(groupby_result, std::cout);
//  print_v(aggregation_result, std::cout);
//
//  // Make sure results are sorted
//  if(1 == flag_sort_result){
//    std::map<int64_t, double> expected_results { {1,3.}, {2,5.}, {3,6.}, {4,7.}, {42, 51.} };
//    ASSERT_EQ(expected_results.size(), gdf_groupby_result.size);
//    ASSERT_EQ(expected_results.size(), gdf_aggregation_result.size);
//
//    int i = 0;
//    for(auto kv : expected_results){
//      EXPECT_EQ(kv.first, groupby_result[i]) << "index: " << i;
//      EXPECT_EQ(kv.second, aggregation_result[i++]) << "index: " << i;
//    }
//  }
//  else
//  {
//    std::unordered_map<int64_t, double> expected_results { {1,3.}, {2,5.}, {3,6.}, {4,7.} };
//    ASSERT_EQ(expected_results.size(), gdf_groupby_result.size);
//    ASSERT_EQ(expected_results.size(), gdf_aggregation_result.size);
//
//    for(int i = 0; i < gdf_aggregation_result.size; ++i){
//      const int64_t key = groupby_result[i];
//      const double value = aggregation_result[i];
//      auto found = expected_results.find(groupby_result[i]);
//      EXPECT_EQ(found->first, key);
//      EXPECT_EQ(found->second, value);
//    }
//  }
//}


