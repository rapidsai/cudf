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

#pragma once

#include <cudf/cudf.h>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>

template <typename T, typename InputIterator>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns(cudf::size_type start, cudf::size_type size, InputIterator valids) {
    auto iter = cudf::test::make_counting_transform_iterator(start, [](auto i) { return T(i);});

    return cudf::test::fixed_width_column_wrapper<T> (iter, iter + size, valids);

}

template <typename T, typename InputIterator>
cudf::experimental::table create_fixed_table(cudf::size_type num_cols, cudf::size_type start, cudf::size_type col_size, InputIterator valids) {        
    std::vector<std::unique_ptr<cudf::column>> cols;    
    for(int idx=0; idx<num_cols; idx++){
        cudf::test::fixed_width_column_wrapper<T> wrap = create_fixed_columns<T>(start + (idx * num_cols), col_size, valids);
        cols.push_back(wrap.release());
    }    
    return cudf::experimental::table(std::move(cols));    
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(std::vector<cudf::size_type> const& indices, bool nullable) {

    std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

    for(unsigned long index = 0; index < indices.size(); index+=2) {
        auto iter = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return T(i);});

        if(not nullable) {
            result.push_back(cudf::test::fixed_width_column_wrapper<T> (iter, iter + (indices[index+1] - indices[index])));
        } else {
            auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return i%2==0? true:false; });
            result.push_back(cudf::test::fixed_width_column_wrapper<T> (iter, iter + (indices[index+1] - indices[index]), valids));
        }
    }

    return result;
}

template <typename T>
std::vector<cudf::experimental::table> create_expected_tables(cudf::size_type num_cols, std::vector<cudf::size_type> const& indices, bool nullable) {

    std::vector<cudf::experimental::table> result;

    for(unsigned long index = 0; index < indices.size(); index+=2) {
        std::vector<std::unique_ptr<cudf::column>> cols = {};
        
        for(int idx=0; idx<num_cols; idx++){            
            auto iter = cudf::test::make_counting_transform_iterator(indices[index] + (idx * num_cols), [](auto i) { return T(i);});

            if(not nullable) {                
                cudf::test::fixed_width_column_wrapper<T> wrap(iter, iter + (indices[index+1] - indices[index]));
                cols.push_back(wrap.release());
            } else {               
                auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return i%2==0? true:false; });
                cudf::test::fixed_width_column_wrapper<T> wrap(iter, iter + (indices[index+1] - indices[index]), valids);
                cols.push_back(wrap.release());
            }
        }

        result.push_back(cudf::experimental::table(std::move(cols)));
    }

    return result;
}

inline std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns(std::vector<std::string> const& strings, std::vector<cudf::size_type> const& indices, bool nullable) {

    std::vector<cudf::test::strings_column_wrapper> result = {};

    for(unsigned long index = 0; index < indices.size(); index+=2) {
        if(not nullable) {
            result.push_back(cudf::test::strings_column_wrapper (strings.begin()+indices[index],  strings.begin()+indices[index+1]));
        } else {
            auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return i%2==0? true:false; });
            result.push_back(cudf::test::strings_column_wrapper (strings.begin()+indices[index], strings.begin()+indices[index+1], valids));
        }
    }

    return result;
}

inline std::vector<cudf::experimental::table> create_expected_string_tables(std::vector<std::string> const strings[2], std::vector<cudf::size_type> const& indices, bool nullable) {

    std::vector<cudf::experimental::table> result = {};

    for(unsigned long index = 0; index < indices.size(); index+=2) {
        std::vector<std::unique_ptr<cudf::column>> cols = {};
        
        for(int idx=0; idx<2; idx++){     
            if(not nullable) {
                cudf::test::strings_column_wrapper wrap(strings[idx].begin()+indices[index], strings[idx].begin()+indices[index+1]);
                cols.push_back(wrap.release());
            } else {
                auto valids = cudf::test::make_counting_transform_iterator(indices[index], [](auto i) { return i%2==0? true:false; });
                cudf::test::strings_column_wrapper wrap(strings[idx].begin()+indices[index], strings[idx].begin()+indices[index+1], valids);
                cols.push_back(wrap.release());
            }
        }

        result.push_back(cudf::experimental::table(std::move(cols)));
    }

    return result;
}