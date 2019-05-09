/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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

#ifndef TABLE_ROWWISE_OPERATIONS_CUH
#define TABLE_ROWWISE_OPERATIONS_CUH

#include <cudf.h>
#include "table/device_table.cuh"

namespace {
enum class State {False = 0, True = 1, Undecided = 2};







// namespace {
// struct elements_are_equal_to_value {
//   template <typename ColumnType>
//   __device__ __forceinline__ bool operator()(gdf_column const& lhs,
//                                              gdf_size_type lhs_index,
//                                              void const* value) {
    
//     return static_cast<ColumnType const*>(lhs.data)[lhs_index] ==
//            static_cast<ColumnType const*>(value)[0];    
//   }
// };
// }  // namespace


// struct equality_to_value_comparator {

//   equality_to_value_comparator(device_table const& lhs, const void *const *vals) :
//                             _lhs(lhs), _vals(vals) {
//   }
  
//   __device__ inline bool operator()(gdf_index_type lhs_index, gdf_index_type rhs_index) {

//     auto equal_elements = [lhs_index, this](
//                             gdf_column const& l, int i) {
//     return cudf::type_dispatcher(l.dtype, elements_are_equal_to_value{}, l, lhs_index, this->_vals[i]);
//     };

//     thrust::counting_iterator<int> iter(0);
//     return thrust::equal(thrust::seq, _lhs.begin(), _lhs.end(), iter.begin(),
//                         equal_elements);
//   }

//   private:
//     device_table const _lhs;
//     const void *const * _vals;    
    
// };




struct typed_inequality_comparator {
  template<typename ColType>
    __device__
    State operator() (gdf_index_type lhs_row, gdf_index_type rhs_row,
                    gdf_column const* lhs_column, gdf_column const* rhs_column,
                    bool nulls_are_smallest)
    {
        const ColType lhs_data = static_cast<const ColType*>(lhs_column->data)[lhs_row];
        const ColType rhs_data = static_cast<const ColType*>(rhs_column->data)[rhs_row];
    
        if( lhs_data < rhs_data )
            return State::True;
        else if( lhs_data == rhs_data )
            return State::Undecided;
        else
            return State::False;
    }
};

struct typed_inequality_with_nulls_comparator {
template<typename ColType>
    __device__
    State operator() (gdf_index_type lhs_row, gdf_index_type rhs_row,
                    gdf_column const* lhs_column, gdf_column const* rhs_column,
                    bool nulls_are_smallest)
    {
        const ColType lhs_data = static_cast<const ColType*>(lhs_column->data)[lhs_row];
        const ColType rhs_data = static_cast<const ColType*>(rhs_column->data)[rhs_row];
        const bool isValid1 = gdf_is_valid(lhs_column->valid, lhs_row);
        const bool isValid2 = gdf_is_valid(rhs_column->valid, rhs_row);

        if (!isValid2 && !isValid1)
            return State::Undecided;
        else if( isValid1 && isValid2)
        {
            if( lhs_data < rhs_data )
                return State::True;
            else if( lhs_data == rhs_data )
                return State::Undecided;
            else
                return State::False;
        }
        else if (!isValid1 && nulls_are_smallest)
            return  State::True;
        else if (!isValid2 && !nulls_are_smallest)
            return State::True;
        else
            return State::False;
    }
};
} // namespace


struct inequality_comparator {

  inequality_comparator(device_table const& lhs, bool nulls_are_smallest = true, int8_t *const asc_desc_flags = nullptr) :
                            _lhs(lhs), _rhs(lhs), _nulls_are_smallest(nulls_are_smallest), _asc_desc_flags(asc_desc_flags) {
    _has_nulls = _lhs.has_nulls();
  }
  inequality_comparator(device_table const& lhs, device_table const& rhs, 
                                                  bool nulls_are_smallest = true, int8_t *const asc_desc_flags = nullptr) :
                            _lhs(lhs), _rhs(rhs), _nulls_are_smallest(nulls_are_smallest), _asc_desc_flags(asc_desc_flags) {
    _has_nulls = _lhs.has_nulls() || _rhs.has_nulls();
  }

  __device__ inline bool operator()(gdf_index_type lhs_index, gdf_index_type rhs_index) {

    State state = State::Undecided;
    if (_has_nulls) { 
      for(gdf_size_type col_index = 0; col_index < _lhs.num_columns(); ++col_index) {
        gdf_dtype col_type = _lhs.get_column(col_index)->dtype;

        bool asc = _asc_desc_flags == nullptr || _asc_desc_flags[col_index] == GDF_ORDER_ASC;
        
        if (asc){
          state = cudf::type_dispatcher(col_type, typed_inequality_with_nulls_comparator{},
                                          lhs_index, rhs_index,
                                          _lhs.get_column(col_index), _rhs.get_column(col_index),
                                          _nulls_are_smallest);
        } else {
          state = cudf::type_dispatcher(col_type, typed_inequality_with_nulls_comparator{},
                                          rhs_index, lhs_index,
                                          _rhs.get_column(col_index), _lhs.get_column(col_index),
                                          _nulls_are_smallest);
        }
        
        switch( state ) {
          case State::False:
            return false;
          case State::True:
            return true;
          case State::Undecided:
            break;
        }
      }
    } else {
      for(gdf_size_type col_index = 0; col_index < _lhs.num_columns(); ++col_index) {
        gdf_dtype col_type = _lhs.get_column(col_index)->dtype;

        bool asc = _asc_desc_flags == nullptr || _asc_desc_flags[col_index] == GDF_ORDER_ASC;
        
        if (asc){
          state = cudf::type_dispatcher(col_type, typed_inequality_comparator{},
                                          lhs_index, rhs_index,
                                          _lhs.get_column(col_index),_rhs.get_column(col_index),
                                          asc);
        } else {
          state = cudf::type_dispatcher(col_type, typed_inequality_comparator{},
                                          rhs_index, lhs_index,
                                          _rhs.get_column(col_index),_lhs.get_column(col_index),
                                          asc);
        }
        
        switch( state ) {
          case State::False:
            return false;
          case State::True:
            return true;
          case State::Undecided:
            break;
        }
      }
    }
    return false;
  }

  
  private:

    device_table const _lhs;
    device_table const _rhs;
    bool _nulls_are_smallest;
    int8_t *const _asc_desc_flags;
    bool _has_nulls;   

};

#endif