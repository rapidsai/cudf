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

#include <cudf/column/column_view.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {

column_view::column_view(void const* data, data_type type, size_type size,
                         std::unique_ptr<column_view> null_mask,
                         size_type null_count,
                         std::vector<column_view> const& children)
    : _data{data},
      _type{type},
      _size{size},
      _null_mask{null_mask},
      _null_count{null_count},
      _children{children} {} {
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");
  if (size > 0) {
    CUDF_EXPECTS(nullptr != data, "Null data pointer.");
    CUDF_EXPECTS(INVALID != type, "Invalid element type.");
  }

  if (null_count > 0) {
    CUDF_EXPECTS(nullptr != null_mask,
                 "Invalid null mask for non-zero null count.");
  }
};

column_view::column_view(column_view const& other)
    : _data{other._data},
      _type{other._type},
      _size{other._size},
      _null_mask{std::make_unique<column_view>(other.)} {}

}  // namespace cudf

/**---------------------------------------------------------------------------*
 * @brief Creates a bitmask by applying a predicate to the elements of a column.
 *
 * Bit `i` in the output bitmask will be set if `p(input[i])` returns true,
 * else, the bit will be unset.
 *
 * @param input The column to apply the predicate on
 * @param p The predicate to apply
 * @return bit_mask_t* A bitmask whose bits are set iff the predicate returned
 * true on the corresponding elements of `input`.
 *---------------------------------------------------------------------------**/
template <typename Predicate>
bit_mask_t* null_if(gdf_column const& input, Predicate p);

struct is_nan{
    template <typename T>
    bool operator()(gdf_size_type index){
        return is_nan(static_cast<T*>(input.data)[index]);
    }
    gdf_column input;
}