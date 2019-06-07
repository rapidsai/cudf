#include <cassert>
#include <thrust/fill.h>
#include <algorithm>
#include <tuple>

#include "cudf.h"
#include "types.hpp"
#include "copying.hpp"
#include "utilities/nvtx/nvtx_utils.h"
#include "utilities/error_utils.hpp"
#include "groupby/aggregation_operations.hpp"
#include "groupby/hash_groupby.cuh"
#include "groupby/sort/groupby_sort.cuh"
#include "string/nvcategory_util.hpp"
#include "table/device_table.cuh"

#include "groupby_sort.cuh"

#include <groupby.hpp>

namespace{
  /* --------------------------------------------------------------------------*/
  /**
   * @brief Verifies that a set gdf_columns contain non-null data buffers, and are all
   * of the same size.
   *
   *
   * TODO: remove when null support added.
   *
   * Also ensures that the columns do not contain any null values
   *
   * @param[in] first Pointer to first gdf_column in set
   * @param[in] last Pointer to one past the last column in set
   *
   * @returns GDF_DATASET_EMPTY if a column contains a null data buffer,
   * GDF_COLUMN_SIZE_MISMATCH if the columns are not of equal length,
   */
  /* ----------------------------------------------------------------------------*/
  gdf_error verify_columns(gdf_column * cols[], int num_cols)
  {
    GDF_REQUIRE((nullptr != cols[0]), GDF_DATASET_EMPTY);

    gdf_size_type const required_size{cols[0]->size};

    for(int i = 0; i < num_cols; ++i)
    {
      GDF_REQUIRE(nullptr != cols[i], GDF_DATASET_EMPTY);
      GDF_REQUIRE(nullptr != cols[i]->data, GDF_DATASET_EMPTY);
      GDF_REQUIRE(required_size == cols[i]->size, GDF_COLUMN_SIZE_MISMATCH );
    }
    return GDF_SUCCESS;
  }
  
} // anonymous namespace


gdf_error gdf_group_by_sort(gdf_column* in_key_columns[],
                       int num_key_columns,
                       gdf_column* in_aggregation_columns[],
                       int num_aggregation_columns,
                       gdf_agg_op agg_ops[],
                       gdf_column* out_key_columns[],
                       gdf_column* out_aggregation_columns[],
                       gdf_context* options)
{

  // TODO: Remove when single pass multi-agg is implemented
  if(num_aggregation_columns > 1)
    assert(false && "Only 1 aggregation column currently supported.");

  // Ensure inputs aren't null
  if( (0 == num_key_columns)
      || (0 == num_aggregation_columns)
      || (nullptr == in_key_columns)
      || (nullptr == in_aggregation_columns)
      || (nullptr == agg_ops)
      || (nullptr == out_key_columns)
      || (nullptr == out_aggregation_columns)
      || (nullptr == options))
  {
    return GDF_DATASET_EMPTY;
  }

  // Return immediately if inputs are empty
  GDF_REQUIRE(0 != in_key_columns[0]->size, GDF_SUCCESS);
  GDF_REQUIRE(0 != in_aggregation_columns[0]->size, GDF_SUCCESS);

  auto result = verify_columns(in_key_columns, num_key_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  result = verify_columns(in_aggregation_columns, num_aggregation_columns);
  GDF_REQUIRE( GDF_SUCCESS == result, result );

  gdf_error gdf_error_code{GDF_SUCCESS};

  PUSH_RANGE("LIBGDF_GROUPBY_SORT", GROUPBY_COLOR);

  // INSTEAD OF: use options object
  // bool sort_result = false;
  // if( 0 != options->flag_sort_result){
  //   sort_result = true;
  // }

  
  //@TODO:   
  // gdf_dtype aggregation_column_type;
  // // FIXME When the aggregation type is COUNT, use the type of the OUTPUT column
  // // as the type of the aggregation column. This is required as there is a limitation 
  // // hash based groupby implementation where it's assumed the aggregation input column
  // // and output column are the same type
  // if(is_same_functor<count_op, op>::value)
  // {
  //   aggregation_column_type = out_aggregation_columns[0]->dtype;
  // }
  // else
  // {
  //   aggregation_column_type = in_aggregation_columns[0]->dtype;
  // }

  rmm::device_vector<int32_t> sorted_indices;
  gdf_error_code = group_by_sort::gdf_group_by_sort_pre(num_key_columns,
                                              in_key_columns,
                                              options,
                                              sorted_indices);

  GDF_REQUIRE(GDF_SUCCESS == gdf_error_code, gdf_error_code);

  gdf_agg_op op{agg_ops[0]};

  switch(op)
  { 
    case GDF_MIN:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort<min_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   options, sorted_indices);
        break;
      } 
    case GDF_MAX:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort<max_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   options, sorted_indices);
        break;
      } 
    case GDF_SUM:
      {
        gdf_error_code = group_by_sort::gdf_group_by_sort<sum_op>(num_key_columns,
                                                   in_key_columns,
                                                   in_aggregation_columns[0],
                                                   out_key_columns,
                                                   out_aggregation_columns[0],
                                                   options, sorted_indices);
        break;
      } 
    // case GDF_COUNT:
    //   {
    //     gdf_error_code = group_by_sort::gdf_group_by_sort<count_op>(num_key_columns,
    //                                                in_key_columns,
    //                                                in_aggregation_columns[0],
    //                                                out_key_columns,
    //                                                out_aggregation_columns[0],
    //                                                options, sorted_indices);
    //     break;
    //   }  
    default:
      std::cerr << "Unsupported aggregation method for sort-based groupby." << std::endl;
      gdf_error_code = GDF_UNSUPPORTED_METHOD;
  }
  GDF_REQUIRE(GDF_SUCCESS == gdf_error_code, gdf_error_code);

  POP_RANGE();

  return gdf_error_code;
}


namespace cudf {
namespace groupby {
namespace sort {
 
namespace {

  void verify_operators(table const& values, std::vector<operators> const& ops) {
    CUDF_EXPECTS(static_cast<gdf_size_type>(ops.size()) == values.num_columns(),
                "Size mismatch between ops and value columns");
    for (gdf_size_type i = 0; i < values.num_columns(); ++i) {
      // TODO Add more checks here, i.e., can't compute sum of non-arithemtic
      // types
      if ((ops[i] == SUM) and
          (values.get_column(i)->dtype == GDF_STRING_CATEGORY)) {
        CUDF_FAIL(
            "Cannot compute SUM aggregation of GDF_STRING_CATEGORY column.");
      }
    }
  }

}

std::tuple<cudf::table, cudf::table> groupby(cudf::table const &keys,
                                             cudf::table const &values,
                                             std::vector<operators> const& ops,
                                             gdf_context* options) 
{
  auto num_key_columns = keys.num_columns(); 
  auto num_aggregation_columns = values.num_columns();
  auto num_key_rows = keys.num_rows(); 
  auto num_value_rows = values.num_rows(); 

  gdf_column* in_key_columns[keys.num_columns()];
  for (gdf_size_type i = 0; i < keys.num_columns(); i++)
  {
    in_key_columns[i] = (gdf_column*)keys.get_column(i);
  }

  verify_operators(values, ops);

  // Ensure inputs aren't null
  if( (0 == num_key_columns)
      || (0 == num_aggregation_columns)
      || (nullptr == options))
  {
    CUDF_FAIL("GDF_DATASET_EMPTY");
  }

  // Return immediately if inputs are empty
  CUDF_EXPECTS(0 != num_key_rows, "num_key_rows != 0");
  CUDF_EXPECTS(0 != num_value_rows, "num_value_rows != 0");

  rmm::device_vector<int32_t> sorted_indices;
  auto gdf_error_code = group_by_sort::gdf_group_by_sort_pre(num_key_columns,
                                              in_key_columns,
                                              options,
                                              sorted_indices);
  
  CUDF_EXPECTS(GDF_SUCCESS == gdf_error_code, "gdf_group_by_sort_pre error: " + gdf_error_code);

  cudf::table out_key_table;
  cudf::table out_agg_table;

  // gdf_column* out_key_columns[],
    // ncols : len(keys.size())        
    // nrows:  ???

  // gdf_column* out_aggregation_columns[], 
    // ncols : len(values.size())        
    // nrows:  ?? 

  // cudf::table output_keys{cudf::allocate_like(keys, stream)};
  // cudf::table output_values{cudf::allocate_like(values, stream)};

  // auto d_output_keys = device_table::create(output_keys, stream);
  // auto d_output_values = device_table::create(output_values, stream);



  auto allocate_table_like  =  [](table const& t, cudaStream_t stream) {
    std::vector<gdf_column*> columns(t.num_columns());
    std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
                  [stream](gdf_column* out_col, gdf_column const* in_col) {
                    out_col = new gdf_column;
                    *out_col = allocate_like(*in_col,stream);
                    return out_col;
                  });

    return table{columns.data(), static_cast<gdf_size_type>(columns.size())};
  };


  // gdf_size_type const size{new_key_rows};
  //     cudf::test::column_wrapper<int> col0{size};
  //     cudf::test::column_wrapper<float> col1{size};
  //     cudf::test::column_wrapper<double> col2{size};
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudf::table output_keys{allocate_table_like(keys, stream)};
  cudf::table output_values{allocate_table_like(values, stream)};


  // gdf_column* out_key_columns[output_keys.num_columns()];
  // for (gdf_size_type i = 0; i < output_keys.num_columns(); i++) {
  //   out_key_columns[i] = output_keys.get_column(i);
  // }
  

  for (size_t i = 0; i < ops.size(); i++)
  {
    switch(ops[i])
    { 
      case operators::SUM:
        {
          // gdf_error_code = group_by_sort::gdf_group_by_sort<sum_op>(num_key_columns,
          //                                           in_key_columns,
          //                                           values.get_column(i),
          //                                           out_key_columns,
          //                                           output_values.get_column(i),
          //                                           options, 
          //                                           sorted_indices);
          break;
        }
      default: 
        {

        }
    }
  }
  return std::make_tuple(out_key_table, out_agg_table);
}


} // namespace sort
} // namespace groupby
} // namespace cudf 