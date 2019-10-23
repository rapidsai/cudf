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

#ifndef COLUMN_WRAPPER_FACTORY_HPP
#define COLUMN_WRAPPER_FACTORY_HPP

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
 * column_wrapper_factory allows you to pass data and bit initializers just like
 * you do for the underlying column_wrapper, but it has a specialization for
 * nvstring_category columns that will convert generated numeric values into 
 * strings and initialize the column wrapper with a vector of those strings.
 * This, combined with column_wrapper's type-aware comparison function allow
 * using the same tests for numeric and string data with no special handling.
 * See copy_range_test.cpp and apply_boolean_mask_tests.cpp for examples.
 *
 * Example:
 *
 * column_wrapper_factory<T> factory{};
 *
 * column_wrapper<T> src = factory.make(size,
 *                                      [](int row) { return 2 * row; },
 *                                      [](int row) { return true; });
 *
 * If T is an int, src.data contains [0, 2, 4, ...]
 * If T is cudf::nvstring_category, src.data contains [0, 1, 2, ...] which are 
 * indices into an NVCategory containing the keys ["0", "2", "4", ...]
 */
template <typename T>
struct column_wrapper_factory
{
  template<typename DataInitializer>
  column_wrapper<T> make(cudf::size_type size, DataInitializer data_init) {
    return column_wrapper<T>(size,
      [&](cudf::size_type row) { return convert(data_init(row)); });
  }

  template<typename DataInitializer, typename BitInitializer>
  column_wrapper<T> make(cudf::size_type size,
                         DataInitializer data_init, BitInitializer bit_init) {
    return column_wrapper<T>(size,
      [&](cudf::size_type row) { return convert(data_init(row)); },
      bit_init);
  }
protected: 
  T convert(cudf::size_type row) {
    return static_cast<T>(row);
  }
};

template <>
struct column_wrapper_factory<cudf::nvstring_category>
{
  template<typename DataInitializer>
  column_wrapper<cudf::nvstring_category> make(cudf::size_type size,
                                               DataInitializer data_init) {
    std::vector<const char*> strings = generate_strings(size, data_init);
    auto c =  column_wrapper<cudf::nvstring_category>{size, strings.data()};
    destroy_strings(strings);
    return c;
  }

  template<typename DataInitializer, typename BitInitializer>
  column_wrapper<cudf::nvstring_category> make(cudf::size_type size,
                                               DataInitializer data_init,
                                               BitInitializer bit_init) {
    std::vector<const char*> strings = generate_strings(size, data_init);
    auto c =  column_wrapper<cudf::nvstring_category>{size, strings.data(),
                                                      bit_init};
    destroy_strings(strings);
    return c;
  }

protected:
  const char* convert(cudf::size_type row) {
    static int str_width = std::to_string(std::numeric_limits<cudf::size_type>().max()).size();
    
    std::ostringstream convert;
    convert << std::setfill('0') << std::setw(str_width) << row;
    char *s = new char[convert.str().size()+1];
    std::strcpy(s, convert.str().c_str());
    return s; 
  }

  template <typename DataInitializer>
  std::vector<const char*> generate_strings(cudf::size_type size,
                                            DataInitializer data_init)
  {
    std::vector<const char*> strings(size);
    std::generate(strings.begin(), strings.end(), [&, row=0]() mutable {
      return convert(data_init(row++));
    });
    return strings;
  }

  void destroy_strings(std::vector<const char*> strings) {
    std::for_each(strings.begin(), strings.end(), [](const char* x) { 
      delete [] x; 
    });
  }
};

} // namespace test

} // namespace cudf

#endif // COLUMN_WRAPPER_FACTORY
