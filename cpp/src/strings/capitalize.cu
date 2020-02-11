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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/case.hpp>
#include <cudf/utilities/error.hpp>
#include <strings/char_types/is_flags.h>
#include <strings/utilities.hpp>
#include <strings/utilities.cuh>


namespace cudf
{
namespace strings
{
namespace detail
{
namespace { // anonym.
  enum class pass_step : int { SizeOnly = 0, ExecuteOp};

  template<typename modifier_functor,
           pass_step Pass = pass_step::SizeOnly>
  struct case_manip
  {
    //selective construction based on class template parameter
    //
    //for SFINAE to work need a memf template parameter, `p`,
    //which defaults to Pass, in order to make the SFINAE dependent on Pass,
    //which really is the intention here
    //
    //otherwise, no SFINAE is possible, because inside the class
    //Pass is already known (fixed);
    //
    //specialization for ExecuteOp:
    //
    template<pass_step p = Pass>
    case_manip(column_device_view const d_column,
               character_flags_table_type case_flag,
               character_flags_table_type const* d_flags,
               character_cases_table_type const* d_case_table,
               int32_t const* d_offsets,
               char* d_chars,
               typename std::enable_if_t<p == pass_step::ExecuteOp>* = nullptr):
      d_column_(d_column),
      case_flag_(case_flag),
      d_flags_(d_flags),
      d_case_table_(d_case_table),
      d_offsets_(d_offsets),
      d_chars_(d_chars)
    {
    }

    //specialization for SizeOnly:
    //
    template<pass_step p = Pass>
    case_manip(column_device_view const d_column,
               character_flags_table_type case_flag,
               character_flags_table_type const* d_flags,
               character_cases_table_type const* d_case_table,
               typename std::enable_if_t<p != pass_step::ExecuteOp>* = nullptr):
      d_column_(d_column),
      case_flag_(case_flag),
      d_flags_(d_flags),
      d_case_table_(d_case_table)
    {
    }

    //same SFINAE mechanism as the one for cnstr.
    //to specialize operator();
    //specialization for ExecuteOp:
    //
    template<pass_step p = Pass>
    __device__
    int32_t operator()(size_type row_index,
                       typename std::enable_if_t<p == pass_step::ExecuteOp>* = nullptr)
    {
      if( d_column_.is_null(row_index) )
        return 0; // null string

      string_view d_str = d_column_.template element<string_view>(row_index);
      char* d_buffer = nullptr;
      d_buffer = d_chars_ + d_offsets_[row_index];

      for( auto itr = d_str.begin(); itr != d_str.end(); ++itr )
        {
          uint32_t code_point = detail::utf8_to_codepoint(*itr);
          detail::character_flags_table_type flag = code_point <= 0x00FFFF ? d_flags_[code_point] : 0;

          modifier_functor(d_buffer, d_case_table_, case_flag_, code_point, flag);
        }

      return 0;
    }

    //specialization for SizeOnly:
    //
    template<pass_step p = Pass>
    __device__
    int32_t operator()(size_type row_index,
                       typename std::enable_if_t<p != pass_step::ExecuteOp>* = nullptr)
    {
      if( d_column_.is_null(row_index) )
        return 0; // null string
      
      int32_t bytes = 0;
      string_view d_str = d_column_.template element<string_view>(row_index);
      for( auto itr = d_str.begin(); itr != d_str.end(); ++itr )
        {
            uint32_t code_point = detail::utf8_to_codepoint(*itr);
            detail::character_flags_table_type flag = code_point <= 0x00FFFF ? d_flags_[code_point] : 0;
            if( flag & case_flag_ )
            {
              bytes += detail::bytes_in_char_utf8(detail::codepoint_to_utf8(d_case_table_[code_point]));
            }
            else
            {
              bytes += detail::bytes_in_char_utf8(*itr);
            }
        }
        return bytes;
    }
  private:
    column_device_view const d_column_;
    character_flags_table_type case_flag_; // flag to check with on each character
    character_flags_table_type const* d_flags_;
    character_cases_table_type const* d_case_table_;
    int32_t const* d_offsets_;
    char* d_chars_;
  };
         
}//anonym.
}//namespace detail

std::unique_ptr<column> capitalize( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  //TODO:
  //
  return nullptr;//for now
}

std::unique_ptr<column> title( strings_column_view const& strings,
                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  //TODO:
  //
  return nullptr;//for now
}
  
}//namespace strings
}//namespace cudf
