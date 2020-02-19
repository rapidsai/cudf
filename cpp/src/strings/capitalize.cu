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
    case_manip(modifier_functor d_fctr,
               column_device_view const d_column,
               character_flags_table_type case_flag,
               character_flags_table_type const* d_flags,
               character_cases_table_type const* d_case_table,
               int32_t const* d_offsets,
               char* d_chars,
               typename std::enable_if_t<p == pass_step::ExecuteOp>* = nullptr):
      d_functor_(d_fctr),
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
    case_manip(modifier_functor d_fctr,
               column_device_view const d_column,
               character_flags_table_type case_flag,
               character_flags_table_type const* d_flags,
               character_cases_table_type const* d_case_table,
               typename std::enable_if_t<p != pass_step::ExecuteOp>* = nullptr):
      d_functor_(d_fctr),
      d_column_(d_column),
      case_flag_(case_flag),
      d_flags_(d_flags),
      d_case_table_(d_case_table),
      d_offsets_(nullptr),
      d_chars_(nullptr)
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

          d_functor_(d_buffer, d_case_table_, case_flag_, code_point, flag);
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
    modifier_functor d_functor_;
    column_device_view const d_column_;
    character_flags_table_type case_flag_; // flag to check with on each character
    character_flags_table_type const* d_flags_;
    character_cases_table_type const* d_case_table_;
    int32_t const* d_offsets_;
    char* d_chars_;
  };


  struct probe_capitalize
  {
    probe_capitalize(column_device_view const d_column,
                     character_flags_table_type const* d_flags,
                     character_cases_table_type const* d_case_table):
      d_column_(d_column),
      d_flags_(d_flags),
      d_case_table_(d_case_table)
    {  
    }
    
     __device__
     int32_t operator()(size_type idx) const {
      if( d_column_.is_null(idx) )
        return 0; // null string
      
      string_view d_str = d_column_.template element<string_view>(idx);
      int32_t bytes = 0;
      
      for( auto itr = d_str.begin(); itr != d_str.end(); ++itr ) {
        auto the_chr = *itr;
        uint32_t code_point = detail::utf8_to_codepoint(the_chr);
        detail::character_flags_table_type flag = code_point <= 0x00FFFF ? d_flags_[code_point] : 0;
        if( (itr == d_str.begin()) ? IS_LOWER(flag) : IS_UPPER(flag) )
          the_chr = detail::codepoint_to_utf8(d_case_table_[code_point]);
        bytes += detail::bytes_in_char_utf8(the_chr);
      }
      return bytes;
    }
  private:
    column_device_view const d_column_;
    character_flags_table_type const* d_flags_;
    character_cases_table_type const* d_case_table_;
  };

  struct execute_capitalize
  {
    execute_capitalize(column_device_view const d_column,
                       character_flags_table_type const* d_flags,
                       character_cases_table_type const* d_case_table,
                       int32_t const* d_offsets,
                       char* d_chars):
      d_column_(d_column),
      d_flags_(d_flags),
      d_case_table_(d_case_table),
      d_offsets_(d_offsets),
      d_chars_(d_chars)
    {
    }
    
    __device__
    int32_t operator()(size_type idx) {
      if( d_column_.is_null(idx) )
        return 0; // null string
      
      string_view d_str = d_column_.template element<string_view>(idx);
      char* d_buffer = d_chars_ + d_offsets_[idx];
      
      for( auto itr = d_str.begin(); itr != d_str.end(); ++itr ) {
        auto the_chr = *itr;
        uint32_t code_point = detail::utf8_to_codepoint(the_chr);
        detail::character_flags_table_type flag = code_point <= 0x00FFFF ? d_flags_[code_point] : 0;

        if( (itr == d_str.begin()) ? IS_LOWER(flag) : IS_UPPER(flag) )
          the_chr = detail::codepoint_to_utf8(d_case_table_[code_point]);
        d_buffer += detail::from_char_utf8(the_chr, d_buffer);
      }
      return 0;
    }
  private:
    column_device_view const d_column_;
    character_flags_table_type const* d_flags_;
    character_cases_table_type const* d_case_table_;
    int32_t const* d_offsets_;
    char* d_chars_;
  };
         
}//anonym.

template<typename device_probe_functor,
         typename device_modifier_functor>
std::unique_ptr<column> modify_strings( strings_column_view const& strings,
                                        character_flags_table_type case_flag,
                                        device_probe_functor d_probe_fctr,
                                        device_modifier_functor d_modifier_fctr,
                                        rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                        cudaStream_t stream = 0)
{
  auto strings_count = strings.size();
  if( strings_count == 0 )
    return detail::make_empty_strings_column(mr,stream);

  auto execpol = rmm::exec_policy(stream);
  
  auto strings_column = column_device_view::create(strings.parent(),stream);
  auto d_column = *strings_column;

  // copy null mask
  rmm::device_buffer null_mask = copy_bitmask(strings.parent(),stream,mr);
  // get the lookup tables used for case conversion
  auto d_flags = get_character_flags_table();
  auto d_case_table = get_character_cases_table();  

  detail::case_manip<decltype(d_probe_fctr), pass_step::SizeOnly> cprobe{d_probe_fctr,
      d_column,
      case_flag,
      d_flags,
      d_case_table};


  // build offsets column -- calculate the size of each output string
  auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0), cprobe);
  auto offsets_column = detail::make_offsets_child_column(offsets_transformer_itr,
                                                          offsets_transformer_itr+strings_count,
                                                          mr, stream);
  auto offsets_view = offsets_column->view();
  auto d_new_offsets = offsets_view.template data<int32_t>();//not sure why this requires `.template` and the next one (`d_chars = ...`) doesn't

  // build the chars column -- convert characters based on case_flag parameter
  size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
  auto chars_column = strings::detail::create_chars_child_column( strings_count, d_column.null_count(), bytes, mr, stream );
  auto chars_view = chars_column->mutable_view();
  auto d_chars = chars_view.data<char>();

  detail::case_manip<device_modifier_functor, pass_step::ExecuteOp> cmanip{d_modifier_fctr,
      d_column,
      case_flag,
      d_flags,
      d_case_table,
      d_new_offsets,
      d_chars};
  
  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<size_type>(0), strings_count, cmanip);
  
  //
  return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                             d_column.null_count(), std::move(null_mask), stream, mr);
}

}//namespace detail

std::unique_ptr<column> capitalize( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  //TODO:
  //
  auto d_probe_fctr = [] __device__ (char* d_buffer,
                                     detail::character_cases_table_type const* d_case_table,
                                     detail::character_flags_table_type case_flag,
                                     uint32_t code_point,
                                     detail::character_flags_table_type flag){
    //purposely empty; used just to instantiate a sizeOnly `case_manip` that doesn't need a functor
  };

  
  //TODO:
  //
  auto d_modifier_fctr = [] __device__ (char* d_buffer,
                             detail::character_cases_table_type const* d_case_table,
                             detail::character_flags_table_type case_flag,
                             uint32_t code_point,
                             detail::character_flags_table_type flag){
    //TODO:
    //....
  };//nothing for now...

  detail::character_flags_table_type case_flag = IS_LOWER(0xFF);// <- ????? for now; TODO

  return detail::modify_strings(strings,
                                case_flag,
                                d_probe_fctr,
                                d_modifier_fctr,
                                mr);
}

std::unique_ptr<column> title( strings_column_view const& strings,
                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  //TODO:
  //
  auto d_probe_fctr = [] __device__ (char* d_buffer,
                                     detail::character_cases_table_type const* d_case_table,
                                     detail::character_flags_table_type case_flag,
                                     uint32_t code_point,
                                     detail::character_flags_table_type flag){
    //purposely empty; used just to instantiate a sizeOnly `case_manip` that doesn't need a functor
  };

  
  //TODO:
  //
  auto d_modifier_fctr = [] __device__ (char* d_buffer,
                             detail::character_cases_table_type const* d_case_table,
                             detail::character_flags_table_type case_flag,
                             uint32_t code_point,
                             detail::character_flags_table_type flag){
    //TODO:
    //....
  };//nothing for now...

  detail::character_flags_table_type case_flag = IS_LOWER(0xFF);// <- ????? for now; TODO

  return detail::modify_strings(strings,
                                case_flag,
                                d_probe_fctr,
                                d_modifier_fctr,
                                mr);
}
  
}//namespace strings
}//namespace cudf
