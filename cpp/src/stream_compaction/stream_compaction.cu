/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include "copy_if.cuh"
 
namespace {

// Returns true if the mask is true and valid (non-null) for index i
// This is the filter functor for apply_boolean_mask
// Note we use a functor here so we can cast to a bitmask_t __restrict__
// pointer on the host side, which we can't do with a lambda.
template <bool has_data, bool has_nulls>
struct boolean_mask_filter
{
  boolean_mask_filter(gdf_column const & boolean_mask) :
    size{boolean_mask.size},
    data{reinterpret_cast<cudf::bool8 *>(boolean_mask.data)},
    bitmask{reinterpret_cast<bit_mask_t *>(boolean_mask.valid)}
    {}

  __device__ inline 
  bool operator()(gdf_index_type i)
  {
    if (i < size) {
      bool valid = !has_nulls || bit_mask::is_valid(bitmask, i);
      bool is_true = !has_data || (cudf::true_v == data[i]);
      return is_true && valid;
    }
    return false;
  }

  gdf_size_type size;
  cudf::bool8 const * __restrict__ data;
  bit_mask_t const  * __restrict__ bitmask;
};

// Returns true if the valid mask is true for index i
// Note we use a functor here so we can cast to a bitmask_t __restrict__
// pointer on the host side, which we can't do with a lambda.
struct valid_filter
{
  valid_filter(gdf_column const & column) :
    size{column.size},
    bitmask{reinterpret_cast<bit_mask_t *>(column.valid)}
    { CUDF_EXPECTS(nullptr != column.valid, "Null valid bitmask");}

  __device__ inline 
  bool operator()(gdf_index_type i)
  {
    if (i < size) {
      bool valid = bit_mask::is_valid(bitmask, i);
      return valid;
    }
    return false;
  }

  gdf_size_type size;
  bit_mask_t const  * __restrict__ bitmask;
};

}  // namespace

namespace cudf {

/*
 * Filters a column using a column of boolean values as a mask.
 *
 * calls apply_filter() with the `boolean_mask_filter` functor.
 */
gdf_column apply_boolean_mask(gdf_column const &input,
                              gdf_column const &boolean_mask) {
  CUDF_EXPECTS(boolean_mask.dtype == GDF_BOOL8, "Mask must be Boolean type");
  CUDF_EXPECTS(boolean_mask.data != nullptr ||
               boolean_mask.valid != nullptr, "Null boolean_mask");
  // zero-size inputs are OK, but otherwise input size must match mask size
  CUDF_EXPECTS(input.size == 0 || input.size == boolean_mask.size, 
               "Column size mismatch");

  if (boolean_mask.data == nullptr)
    return detail::copy_if(input, boolean_mask_filter<false, true>{boolean_mask});
  else if (boolean_mask.valid == nullptr || boolean_mask.null_count == 0)
    return detail::copy_if(input, boolean_mask_filter<true, false>{boolean_mask});
  else
    return detail::copy_if(input, boolean_mask_filter<true, true>{boolean_mask});
}

/*
 * Filters a column to remove null elements.
 */
gdf_column drop_nulls(gdf_column const &input) {
  if (input.valid != nullptr && input.null_count != 0)
    return detail::copy_if(input, valid_filter{input});
  else { // no null bitmask, so just copy
    return cudf::copy(input);
  }
}

}  // namespace cudf
