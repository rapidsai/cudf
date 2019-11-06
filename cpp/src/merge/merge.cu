
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <type_traits>
#include <nvstrings/NVCategory.h>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/cuda_utils.hpp>

namespace {

/**
 * @brief Source table identifier to copy data from.
 */
enum class side : bool { LEFT, RIGHT };
  
using index_type = thrust::tuple<side, cudf::size_type>; // `thrust::get<0>` indicates left/right side, `thrust::get<1>` indicates the row index

  /*
rmm::device_vector<index_type>
generate_merged_indices(device_table const& left_table,
                        device_table const& right_table,
                        rmm::device_vector<int8_t> const& asc_desc,
                        bool nulls_are_smallest,
                        cudaStream_t stream) {

    const cudf::size_type left_size  = left_table.num_rows();
    const cudf::size_type right_size = right_table.num_rows();
    const cudf::size_type total_size = left_size + right_size;

    thrust::constant_iterator<side> left_side(side::LEFT);
    thrust::constant_iterator<side> right_side(side::RIGHT);

    auto left_indices = thrust::make_counting_iterator(static_cast<cudf::size_type>(0));
    auto right_indices = thrust::make_counting_iterator(static_cast<cudf::size_type>(0));

    auto left_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side, left_indices));
    auto right_begin_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side, right_indices));

    auto left_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(left_side + left_size, left_indices + left_size));
    auto right_end_zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(right_side + right_size, right_indices + right_size));

    rmm::device_vector<index_type> merged_indices(total_size);
    bool nullable = left_table.has_nulls() || right_table.has_nulls();
    auto exec_pol = rmm::exec_policy(stream);
    if (nullable){
        auto ineq_op = row_inequality_comparator<true>(right_table, left_table, nulls_are_smallest, asc_desc.data().get()); 
        thrust::merge(exec_pol->on(stream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    merged_indices.begin(),
                    [=] __device__ (thrust::tuple<side, cudf::size_type> const & right_tuple,
                                    thrust::tuple<side, cudf::size_type> const & left_tuple) {
                        return ineq_op(thrust::get<1>(right_tuple), thrust::get<1>(left_tuple));
                    });			        
    } else {
        auto ineq_op = row_inequality_comparator<false>(right_table, left_table, nulls_are_smallest, asc_desc.data().get()); 
        thrust::merge(exec_pol->on(stream),
                    left_begin_zip_iterator,
                    left_end_zip_iterator,
                    right_begin_zip_iterator,
                    right_end_zip_iterator,
                    merged_indices.begin(),
                    [=] __device__ (thrust::tuple<side, cudf::size_type> const & right_tuple,
                                    thrust::tuple<side, cudf::size_type> const & left_tuple) {
                        return ineq_op(thrust::get<1>(right_tuple), thrust::get<1>(left_tuple));
                    });					        
    }

    CHECK_STREAM(stream);

    return merged_indices;
}
  */

} // namespace

namespace cudf {
namespace experimental { 
namespace detail {

//nope,
//a column-wise merge may not make sense,
//because of key_col, asc_desc
//(see below...)
//
template<typename VectorI>
struct ColumnMerger
{
  explicit ColumnMerger(VectorI const& row_order):
    dv_row_order_(row_order)
  //dv_row_order_ := (device) container giving
  //the order of rows induced by lexicographic sorting of columns in key_cols
  {
  }
  
  // type_dispatcher() _can_ dispatch host functors:
  //
  template<typename Element>//required: column type
  std::unique_ptr<column>
  operator()(cudf::column_view const& lcol, cudf::column_view const& rcol)
  {
    return nullptr;//for now...
  }
private:
  VectorI const& dv_row_order_;
  
  //see `class element_relational_comparator` in `cpp/include/cudf/table/row_operators.cuh` as a model;
};
  

std::unique_ptr<cudf::experimental::table> merge(table_view const& left_table,
                                   table_view const& right_table,
                                   std::vector<cudf::size_type> const& key_cols,
                                   std::vector<cudf::order> const& asc_desc,
                                   std::vector<cudf::null_order> const& null_precedence) {
    auto n_cols = left_table.num_columns();
    CUDF_EXPECTS( n_cols == right_table.num_columns(), "Mismatched number of columns");
    if (left_table.num_columns() == 0) {
        return nullptr;
    }

    // TODO: replace / drop;
    //no replacement, yet:
    //{
    //proposal for a replacement: bool have_same_types(table_view const& lhs, table_view const& rhs);
    //
    //std::vector<gdf_dtype> left_table_dtypes = cudf::column_dtypes(left_table);
    //std::vector<gdf_dtype> right_table_dtypes = cudf::column_dtypes(right_table);
    //CUDF_EXPECTS(std::equal(left_table_dtypes.cbegin(), left_table_dtypes.cend(), right_table_dtypes.cbegin(), right_table_dtypes.cend()), "Mismatched column dtypes");
    //}
    
    CUDF_EXPECTS(key_cols.size() > 0, "Empty key_cols");
    CUDF_EXPECTS(key_cols.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in key_cols");
    CUDF_EXPECTS(asc_desc.size() > 0, "Empty asc_desc");
    CUDF_EXPECTS(asc_desc.size() <= static_cast<size_t>(left_table.num_columns()), "Too many values in asc_desc");
    CUDF_EXPECTS(key_cols.size() == asc_desc.size(), "Mismatched size between key_cols and asc_desc");


    using column_rep_t = cudf::column; // or column_view?
    using col_ptr_t = typename std::unique_ptr<column_rep_t>;

    std::vector<col_ptr_t> v_merged_cols;
    v_merged_cols.reserve(n_cols);

    static_assert(std::is_same<decltype(v_merged_cols), std::vector<std::unique_ptr<cudf::column>> >::value, "ERROR: unexpected type.");

    for(auto i=0;i<n_cols;++i)
      {
        const auto& left_col = left_table.column(i);
        const auto& right_col= right_table.column(i);

        //not clear yet what must be done for STRING:
        //
        //if( left_col.type().id() != STRING )
        //  continue;//?

        ColumnMerger merger{key_cols[i],
                            asc_desc[i],
                            null_precedence[i]};

        col_ptr_t merged = cudf::experimental::type_dispatcher(left_col.type(),
                                                               merger,
                                                               left_col,
                                                               right_col);
        v_merged_cols.emplace_back(std::move(merged));
      }
    
    return std::unique_ptr<cudf::experimental::table>{new cudf::experimental::table(std::move(v_merged_cols))};
}

}  // namespace detail

std::unique_ptr<cudf::experimental::table> merge(table_view const& left_table,
                                                 table_view const& right_table,
                                                 std::vector<cudf::size_type> const& key_cols,
                                                 std::vector<cudf::order> const& asc_desc,
                                                 std::vector<cudf::null_order> const& null_precedence){
  return detail::merge(left_table, right_table, key_cols, asc_desc, null_precedence);
}

}  // namespace experimental
}  // namespace cudf
