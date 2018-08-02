#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <type_traits>
#include <memory>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

#include "../../joining.h"

// See this header for all of the recursive handling of tuples of vectors
#include "tuple_vectors.h"

// Selects the kind of join operation that is performed
enum struct join_kind
{
  INNER,
  LEFT
};

// TODO Can probably just replace this with a std::pair
// Each element of the result will be an index into the left and right columns where
// left_columns[left_index] == right_columns[right_index]
struct result_type 
{
  size_t left_index{};
  size_t right_index{};

  result_type() {}

  result_type(size_t _l, size_t _r) : 
    left_index{_l}, right_index{_r} {}

  // Overload comparison so the result vector can be sorted
  bool operator <(result_type const& rhs){
    return( std::tie(left_index, right_index) < std::tie(rhs.left_index, rhs.right_index) );
  }

  bool operator ==(result_type const& rhs) const{
    return ( std::tie(left_index, right_index) == std::tie(rhs.left_index, rhs.right_index) );
  }

  friend std::ostream& operator<<(std::ostream& os, const result_type& result);

};

// Overload the stream operator to make it easier to print a result 
std::ostream& operator<<(std::ostream& os, const result_type& result)
{
  os << result.left_index << ", " << result.right_index << std::endl;
  return os;
}

// A new instance of this class will be created for each *TEST(JoinTest, ...)
// Put all repeated setup and validation stuff here
template <class test_parameters>
struct JoinTest : public testing::Test
{
  const join_kind join_method = test_parameters::join_method;

  // multi_column_t is a tuple of vectors. The number of vectors in the tuple
  // determines the number of columns to be joined, and the value_type of each
  // vector determiens the data type of the column
  using multi_column_t = typename test_parameters::multi_column_t;
  multi_column_t left_columns;
  multi_column_t right_columns;

  // Type for a unique_ptr to a gdf_column with a custom deleter
  // Custom deleter is defined at construction
  using gdf_col_pointer = typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

  // Containers for unique_ptrs to gdf_columns that will be used in the gdf_join functions
  // unique_ptrs are used to automate freeing device memory
  std::vector<gdf_col_pointer> gdf_left_columns;
  std::vector<gdf_col_pointer> gdf_right_columns;

  // Containers for the raw pointers to the gdf_columns that will be used as input
  // to the gdf_join functions
  std::vector<gdf_column*> gdf_raw_left_columns;
  std::vector<gdf_column*> gdf_raw_right_columns;

  JoinTest()
  {
    // Use constant seed so the psuedo-random order is the same each time
    // Each time the class is constructed a new constant seed is used
    static size_t number_of_instantiations{0};
    std::srand(number_of_instantiations++);
  }


  ~JoinTest()
  {
  }

    /* --------------------------------------------------------------------------*/
    /** 
     * @Synopsis  Creates a unique_ptr that wraps a gdf_column structure intialized with a host vector
     * 
     * @Param host_vector The host vector whose data is used to initialize the gdf_column
     * 
     * @Returns A unique_ptr wrapping the new gdf_column
     */
    /* ----------------------------------------------------------------------------*/
  template <typename col_type>
  gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector)
  {
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

    // Create a new instance of a gdf_column with a custom deleter that will free
    // the associated device memory when it eventually goes out of scope
    auto deleter = [](gdf_column* col){col->size = 0; cudaFree(col->data);};
    gdf_col_pointer the_column{new gdf_column, deleter};

    // Allocate device storage for gdf_column and copy contents from host_vector
    cudaMalloc(&(the_column->data), host_vector.size() * sizeof(col_type));
    cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice);

    // Fill the gdf_column members
    the_column->valid = nullptr;
    the_column->size = host_vector.size();
    the_column->dtype = gdf_col_type;
    gdf_dtype_extra_info extra_info;
    extra_info.time_unit = TIME_UNIT_NONE;
    the_column->dtype_info = extra_info;

    return the_column;
  }

  // Compile time recursion to convert each vector in a tuple of vectors into
  // a gdf_column and append it to a vector of gdf_columns
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t)
  {
    //bottom of compile-time recursion
    //purposely empty...
  }
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  convert_tuple_to_gdf_columns(std::vector<gdf_col_pointer> &gdf_columns,std::tuple<std::vector<Tp>...>& t)
  {
    // Creates a gdf_column for the current vector and pushes it onto 
    // the vector of gdf_columns
    gdf_columns.push_back(create_gdf_column(std::get<I>(t)));

    //recurse to next vector in tuple
    convert_tuple_to_gdf_columns<I + 1, Tp...>(gdf_columns, t);
  }

  // Converts a tuple of host vectors into a vector of gdf_columns
  std::vector<gdf_col_pointer> 
  initialize_gdf_columns(multi_column_t host_columns)
  {
    std::vector<gdf_col_pointer> gdf_columns;
    convert_tuple_to_gdf_columns(gdf_columns, host_columns);
    return gdf_columns;
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Initializes two sets of columns, left and right, with random values for the join operation.
   * 
   * @Param left_column_length The length of the left set of columns
   * @Param left_column_range The upper bound of random values for the left columns. Values are [0, left_column_range)
   * @Param right_column_length The length of the right set of columns
   * @Param right_column_range The upper bound of random values for the right columns. Values are [0, right_column_range)
   * @Param print Optionally print the left and right set of columns for debug
   */
  /* ----------------------------------------------------------------------------*/
  void create_input( size_t left_column_length, size_t left_column_range,
                     size_t right_column_length, size_t right_column_range,
                     bool print = false)
  {
    initialize_tuple(left_columns, left_column_length, left_column_range); 
    initialize_tuple(right_columns, right_column_length, right_column_range); 

    gdf_left_columns = initialize_gdf_columns(left_columns);
    gdf_right_columns = initialize_gdf_columns(right_columns);

    // Fill vector of raw pointers to gdf_columns
    for(auto const& c : gdf_left_columns){
      gdf_raw_left_columns.push_back(c.get());
    }

    for(auto const& c : gdf_right_columns){
      gdf_raw_right_columns.push_back(c.get());
    }

    if(print)
    {
      std::cout << "Left column(s) created. Size: " << std::get<0>(left_columns).size() << std::endl;
      print_tuple(left_columns);

      std::cout << "Right column(s) created. Size: " << std::get<0>(right_columns).size() << std::endl;
      print_tuple(right_columns);
    }
  }

  
  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Computes a reference solution for joining the left and right sets of columns
   * 
   * @Param print Option to print the solution for debug
   * @Param sort Option to sort the solution. This is necessary for comparison against the gdf solution
   * 
   * @Returns A vector of 'result_type' where result_type is a structure with a left_index, right_index
   * where left_columns[left_index] == right_columns[right_index]
   */
  /* ----------------------------------------------------------------------------*/
  std::vector<result_type> compute_reference_solution(bool print = false, bool sort = true)
  {

    // Use the type of the first vector as the key_type
    using key_type = typename std::tuple_element<0, multi_column_t>::type::value_type;
    using value_type = size_t;

    // Multimap used to compute the reference solution
    std::multimap<key_type, value_type> the_map;

    // Build hash table that maps the first right columns' values to their row index in the column
    std::vector<key_type> const & build_column = std::get<0>(right_columns);
    for(size_t right_index = 0; right_index < build_column.size(); ++right_index)
    {
      the_map.insert(std::make_pair(build_column[right_index], right_index));
    }

    std::vector<result_type> result;

    // Probe hash table with first left column
    std::vector<key_type> const & probe_column = std::get<0>(left_columns);
    for(size_t left_index = 0; left_index < probe_column.size(); ++left_index)
    {
      // Find all keys that match probe_key
      const auto probe_key = probe_column[left_index];
      auto range = the_map.equal_range(probe_key);

      // Every element in the returned range identifies a row in the first right column that
      // matches the probe_key. Need to check if all other columns also match
      bool match{false};
      for(auto i = range.first; i != range.second; ++i)
      {
        const auto right_index = i->second;

        // If all of the columns in right_columns[right_index] == all of the columns in left_columns[left_index]
        // Then this index pair is added to the result as a matching pair of row indices
        if( true == rows_equal(left_columns, right_columns, left_index, right_index)){
          result.emplace_back(left_index, right_index);
          match = true;
        }
      }

      // For left joins, insert a NULL if no match is found
      if((false == match) && (join_method == join_kind::LEFT)){
        constexpr int JoinNullValue{-1};
        result.emplace_back(left_index, JoinNullValue);
      }
    }

    // Sort the result
    if(sort)
    {
      std::sort(result.begin(), result.end());
    }

    if(print)
    {
      std::cout << "Reference result size: " << result.size() << std::endl;
      std::cout << "left index, right index" << std::endl;
      std::copy(result.begin(), result.end(), std::ostream_iterator<result_type>(std::cout, ""));
      std::cout << "\n";
    }

    return result;
  }

  /* --------------------------------------------------------------------------*/
  /** 
   * @Synopsis  Computes the result of joining the left and right sets of columns with the libgdf functions
   * 
   * @Param gdf_result A vector of result_type that holds the result of the libgdf join function
   * @Param print Option to print the result computed by the libgdf function
   * @Param sort Option to sort the result. This is required to compare the result against the reference solution
   */
  /* ----------------------------------------------------------------------------*/
  void compute_gdf_result(std::vector<result_type> & gdf_result, bool print = false, bool sort = true)
  {

    const int num_columns = std::tuple_size<multi_column_t>::value;

    // TODO Don't forget to free the result!
    gdf_join_result_type * gdf_join_result;

    gdf_error result_error{GDF_SUCCESS};

    // Use single column join when there's only a single column
    if(num_columns == 1){
      gdf_column * left_gdf_column = gdf_raw_left_columns[0];
      gdf_column * right_gdf_column = gdf_raw_right_columns[0];
      switch(join_method)
      {
        case join_kind::LEFT:
          {
            result_error = gdf_left_join_generic(left_gdf_column,
                                                 right_gdf_column,
                                                 &gdf_join_result);
            break;
          }
        case join_kind::INNER:
          {
            result_error = gdf_inner_join_generic(left_gdf_column,
                                                  right_gdf_column,
                                                  &gdf_join_result);
            break;
          }
        default:
          std::cout << "Invalid join method" << std::endl;
          ASSERT_TRUE(false);
      }

    }
    // Otherwise use the multicolumn join
    else
    {
      gdf_column ** left_gdf_columns = gdf_raw_left_columns.data();
      gdf_column ** right_gdf_columns = gdf_raw_right_columns.data();
      switch(join_method)
      {
        case join_kind::LEFT:
          {
            result_error = gdf_multi_left_join_generic(num_columns,
                                                       left_gdf_columns,
                                                       right_gdf_columns,
                                                       &gdf_join_result);
            break;
          }
        case join_kind::INNER:
          {
            //result_error =  gdf_multi_inner_join_generic(num_columns,
            //                                             left_gdf_columns, 
            //                                             right_gdf_columns, 
            //                                             &gdf_join_result);
            std::cout << "Multi column *inner* joins not supported yet\n";
            ASSERT_TRUE(false);
            break;
          }
        default:
          std::cout << "Invalid join method" << std::endl;
          ASSERT_TRUE(false);
      }
    }
    ASSERT_EQ(GDF_SUCCESS, result_error) << "The gdf join function did not complete successfully";

    // The output is an array of size `n` where the first n/2 elements are the 
    // left_indices and the last n/2 elements are the right indices
    size_t output_size = gdf_join_result_size(gdf_join_result);
    size_t total_pairs = output_size/2;

    // Copy join result to a Thrust vector
    int * join_output = static_cast<int*>(gdf_join_result_data(gdf_join_result));
    thrust::device_vector<int> d_output(join_output, join_output + output_size);

    // Free the original join result
    gdf_join_result_free(gdf_join_result);

    thrust::device_vector<int> left_indices(d_output.begin(), d_output.begin() + total_pairs);
    thrust::device_vector<int> right_indices(d_output.begin() + total_pairs, d_output.end());

    gdf_result.resize(total_pairs);

    for(size_t i = 0; i < total_pairs; ++i){
      gdf_result[i].left_index = left_indices[i];
      gdf_result[i].right_index = right_indices[i];
    }

    // Sort the output for comparison to reference solution
    if(sort){
      std::sort(gdf_result.begin(), gdf_result.end());
    }

    if(print){
      std::cout << "GDF result size: " << gdf_result.size() << std::endl;
      std::cout << "left index, right index" << std::endl;
      std::copy(gdf_result.begin(), gdf_result.end(), std::ostream_iterator<result_type>(std::cout, ""));
      std::cout << "\n";
    }
  }


  //gdf_error gdf_multi_left_join_generic(int num_cols, gdf_column **leftcol, gdf_column **rightcol, gdf_join_result_type **out_result)
};

// This structure is used to nest the join method and number/types of columns
// for use with Google Test type-parameterized tests
template<join_kind join_type, typename tuple_type>
struct TestParameters
{
  // The method to use for the join
  const static join_kind join_method{join_type};

  // The tuple of vectors that determines the number and types of the columns to join
  using multi_column_t = tuple_type;
};


// Every test defined as TYPED_TEST(JoinTest, *) will be run once for every instance of
// TestParameters defined below
typedef ::testing::Types< 
                          // Single column inner join tests for all types
                          TestParameters< join_kind::INNER, std::tuple<std::vector<int32_t>> >,
                          TestParameters< join_kind::INNER, std::tuple<std::vector<int64_t>> >,
                          TestParameters< join_kind::INNER, std::tuple<std::vector<float>> >,
                          TestParameters< join_kind::INNER, std::tuple<std::vector<double>> >,
                          TestParameters< join_kind::INNER, std::tuple<std::vector<uint32_t>> >,
                          TestParameters< join_kind::INNER, std::tuple<std::vector<uint64_t>> >,
                          // Single column left join tests for all types
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<int32_t>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<int64_t>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<float>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<double>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<uint32_t>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<uint64_t>> >,
                          // Two Column Left Join tests for some combination of types
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<int32_t>, std::vector<int32_t>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<int64_t>, std::vector<int32_t>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<float>, std::vector<double>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<double>, std::vector<int64_t>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<uint32_t>, std::vector<int32_t>> >,
                          // Three Column Left Join tests for some combination of types
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<int32_t>, std::vector<uint32_t>, std::vector<float>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<uint64_t>, std::vector<uint32_t>, std::vector<float>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<float>, std::vector<double>, std::vector<float>> >,
                          TestParameters< join_kind::LEFT, std::tuple<std::vector<double>, std::vector<uint32_t>, std::vector<int64_t>> >
                          > Implementations;

TYPED_TEST_CASE(JoinTest, Implementations);

TYPED_TEST(JoinTest, ExampleTest)
{

  this->create_input(1000,10,
                     1000,10);

  std::vector<result_type> reference_result = this->compute_reference_solution();

  std::vector<result_type> gdf_result{};
  this->compute_gdf_result(gdf_result);

  ASSERT_EQ(reference_result.size(), gdf_result.size()) << "Size of gdf result does not match reference result\n";

  for(size_t i = 0; i < reference_result.size(); ++i){
    EXPECT_EQ(reference_result[i], gdf_result[i]);
  }
}
