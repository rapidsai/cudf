/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 William Malpica <william@blazingdb.com>
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

#ifndef MAKE_COLUMN_WRAPPER_HPP
#define MAKE_COLUMN_WRAPPER_HPP

#include "column_wrapper.cuh"

namespace cudf {

namespace test {

/**
 * @brief Convenience factory for column_wrapper that can generate columns of
 * any cudf datatype from simple integer generators.
 * 
 * Often we need to write gtests for a variety of cudf data types. 
 * string_category columns, for example, often require a separate path 
 * from numeric data types when generating input and reference columns and
 * running tests.
 * make_column_wrapper allows you to pass data and bit initializers just like
 * you do for the underlying column_wrapper, but it has a specialization for
 * nvstring_category columns that will convert generated numeric values into 
 * strings and initialize the column wrapper with a vector of those strings.
 * This, combined with column_wrapper's type-aware comparison function allow
 * using the same tests for numeric and string data with no special handling.
 * See copy_range_test.cpp and apply_boolean_mask_tests.cpp for examples.
 * 
 * Example:
 * 
 * make_column_wrapper<T> maker{};
 * 
 * column_wrapper<T> src = make(size, [](int row) { return 2 * row; },
 *                                    [](int row) { return true; });
 * 
 * If T is an int, src.data contains [0, 2, 4, ...]
 * If T is cudf::nvstring_category, src.data contains [0, 1, 2, ...] which are 
 * indices into an NVCategory containing the keys ["0", "2", "4", ...]
 */
template <typename T>
struct make_column_wrapper
{
  T convert(gdf_index_type row) {
    return static_cast<T>(row);
  }

  template<typename DataInitializer>
  column_wrapper<T>
  operator()(gdf_size_type size,
             DataInitializer data_init) {
    return column_wrapper<T>(size,
      [&](gdf_index_type row) { return convert(data_init(row)); });
  }

  template<typename DataInitializer, typename BitInitializer>
  column_wrapper<T>
  operator()(gdf_size_type size,
             DataInitializer data_init,
             BitInitializer bit_init) {
    return column_wrapper<T>(size,
      [&](gdf_index_type row) { return convert(data_init(row)); },
      bit_init);
  }
};

template <>
struct make_column_wrapper<cudf::nvstring_category>
{
  int scale;

  const char* convert(gdf_index_type row) {
    std::ostringstream convert;
    convert << row;
    char *s = new char[convert.str().size()+1];
    std::strcpy(s, convert.str().c_str());
    return s; 
  }

  template<typename DataInitializer>
  column_wrapper<cudf::nvstring_category>
  operator()(gdf_size_type size,
             DataInitializer data_init) {
    std::vector<const char*> strings(size);
    std::generate(strings.begin(), strings.end(), [&, row=0]() mutable {
      return convert(data_init(row++));
    });
    
    auto c =  column_wrapper<cudf::nvstring_category>{size, strings.data()};
    
    std::for_each(strings.begin(), strings.end(), [](const char* x) { 
      delete [] x; 
    });

    return c;
  }

  template<typename DataInitializer, typename BitInitializer>
  column_wrapper<cudf::nvstring_category>
  operator()(gdf_size_type size,
             DataInitializer data_init,
             BitInitializer bit_init) {
    std::vector<const char*> strings(size);
    std::generate(strings.begin(), strings.end(), [&, row=0]() mutable {
      return convert(data_init(row++));
    });

    auto c =  column_wrapper<cudf::nvstring_category>{size,
                                                      strings.data(),
                                                      bit_init};
    
    std::for_each(strings.begin(), strings.end(), [](const char* x) { 
      delete [] x; 
    });

    return c;
  }
};

} // namespace test

} // namespace cudf

#endif // MAKE_COLUMN_WRAPPER