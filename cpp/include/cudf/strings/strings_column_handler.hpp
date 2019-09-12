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

#include <cudf/column/column_view.hpp>

#include <vector>

namespace cudf {

class strings_column_handler
{
 public:
  ~strings_column_handler() = default;

  strings_column_handler( const column_view& strings_column );
  //strings_column_handler( const column_view&& strings_column );

  size_type count() const;

  const char* chars_data() const;
  const int32_t* offsets_data() const;

  size_type chars_column_size() const;

  const bitmask_type* null_mask() const;
  size_type null_count() const;

  enum sort_type {
        none=0,    ///< no sorting
        length=1,  ///< sort by string length
        name=2     ///< sort by characters code-points
    };

  // print strings to stdout
  void print( size_type start=0, size_type end=-1,
              size_type max_width=-1, const char* delimiter = "\n" ) const;

  // new strings column from subset of given strings column
  std::unique_ptr<cudf::column> sublist( size_type start, size_type end, size_type step );

  // return sorted version of the given strings column
  std::unique_ptr<cudf::column> sort( sort_type stype, bool ascending=true, bool nullfirst=true );

  // return sorted indexes only -- returns integer column
  std::unique_ptr<cudf::column> order( sort_type stype, bool ascending, bool nullfirst=true );

private:
  const column_view _parent;
};

}
