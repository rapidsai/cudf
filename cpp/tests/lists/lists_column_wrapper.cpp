/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <initializer_list>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/concatenate.hpp>

namespace cudf {
namespace test {

std::unique_ptr<column> list_column_wrapper::build_wrapper(std::initializer_list<list_column_wrapper> t, std::vector<bool> const& v)
{
   auto valids = cudf::test::make_counting_transform_iterator(0, [&v](auto i) { return v.size() <= 0 ? true : v[i]; });

   // generate offsets column and do some type checking to make sure the user hasn't passed an invalid initializer list
   type_id child_id = EMPTY;
   size_type count = 0;
   std::vector<size_type> offsetv;
   std::transform(t.begin(), t.end(), valids, std::back_inserter(offsetv), [&](list_column_wrapper const& l, bool valid){
      // verify all children are of the same type (C++ allows you to use initializer
      // lists that could construct an invalid list column type)
      if(child_id == EMPTY){
         child_id = l.wrapped->type().id();
      } else {
         CUDF_EXPECTS(child_id == l.wrapped->type().id(), "Mismatched list types");
      }

      // nulls are represented as a repeated offset
      size_type ret = count; 
      if(valid){
         count += l.wrapped->size();
      }
      return ret;         
   });
   // add the final offset
   offsetv.push_back(count);       
   auto offsets = cudf::test::fixed_width_column_wrapper<size_type>(offsetv.begin(), offsetv.end()).release();

   // concatenate them together, skipping data for children that are null
   std::vector<column_view> children;
   auto l = t.begin();
   for(int idx=0; idx<t.size(); idx++, l++){
      CUDF_EXPECTS(l->wrapped->type().id() == child_id, "Mismatched list types");
      if(valids[idx]){
         children.push_back(*(l->wrapped));
      }   
   }
   auto data = concatenate(children);

   // construct the list column
   return make_lists_column(t.size(), std::move(offsets), std::move(data), 
                               v.size() <= 0 ? 0 : cudf::UNKNOWN_NULL_COUNT,
                               v.size() <= 0 ? rmm::device_buffer{0} : detail::make_null_mask(v.begin(), v.end()));
}

}  // namespace test
}  // namespace cudf