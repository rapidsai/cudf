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
  
  //base class for probing string
  //manipulation memory load requirements;
  //
  struct probe_base
  {
    probe_base(column_device_view const d_column,
               character_flags_table_type const* d_flags,
               character_cases_table_type const* d_case_table):
      d_column_(d_column),
      d_flags_(d_flags),
      d_case_table_(d_case_table)
    {  
    }

    __host__ __device__
    column_device_view const get_column(void) const
    {
      return d_column_;
    }

    __host__ __device__
    character_flags_table_type const* get_flags(void) const
    {
      return d_flags_;
    }

    __host__ __device__
    character_cases_table_type const* get_case_table(void) const
    {
      return d_case_table_;
    }

    __device__
    virtual int32_t operator()(size_type idx) const = 0;
    
  private:
    column_device_view const d_column_;
    character_flags_table_type const* d_flags_;
    character_cases_table_type const* d_case_table_;
  };

  
  //base class for executing string modification:
  //
  struct execute_base
  {
    execute_base(column_device_view const d_column,
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

    __host__ __device__
    column_device_view const get_column(void) const
    {
      return d_column_;
    }

    __host__ __device__
    character_flags_table_type const* get_flags(void) const
    {
      return d_flags_;
    }

    __host__ __device__
    character_cases_table_type const* get_case_table(void) const
    {
      return d_case_table_;
    }

    __host__ __device__
    int32_t const* get_offsets(void) const
    {
      return d_offsets_;
    }

    __host__ __device__
    char* get_chars(void)
    {
      return d_chars_;
    }

    __device__
    virtual int32_t operator()(size_type idx) = 0;
  private:
    column_device_view const d_column_;
    character_flags_table_type const* d_flags_;
    character_cases_table_type const* d_case_table_;
    int32_t const* d_offsets_;
    char* d_chars_;
  };


  //functor for probing string capitalization
  //requirements:
  //
  struct probe_capitalize: probe_base
  {
    probe_capitalize(column_device_view const d_column,
                     character_flags_table_type const* d_flags,
                     character_cases_table_type const* d_case_table):
      probe_base(d_column, d_flags, d_case_table)
    {  
    }
    
     __device__
     int32_t operator()(size_type idx) const override {
       if( get_column().is_null(idx) )
         return 0; // null string
      
       string_view d_str = get_column().template element<string_view>(idx);
       int32_t bytes = 0;
      
       for( auto itr = d_str.begin(); itr != d_str.end(); ++itr ) {
         auto the_chr = *itr;
         uint32_t code_point = detail::utf8_to_codepoint(the_chr);
         detail::character_flags_table_type flag = code_point <= 0x00FFFF ? get_flags()[code_point] : 0;
         if( (itr == d_str.begin()) ? IS_LOWER(flag) : IS_UPPER(flag) )
           the_chr = detail::codepoint_to_utf8(get_case_table()[code_point]);
         bytes += detail::bytes_in_char_utf8(the_chr);
       }
       return bytes;
    }
  };

  //functor for executing string capitalization:
  //
  struct execute_capitalize: execute_base
  {
    execute_capitalize(column_device_view const d_column,
                       character_flags_table_type const* d_flags,
                       character_cases_table_type const* d_case_table,
                       int32_t const* d_offsets,
                       char* d_chars):
      execute_base(d_column, d_flags, d_case_table, d_offsets, d_chars)
    {
    }
    
    __device__
    int32_t operator()(size_type idx) override {
      if( get_column().is_null(idx) )
        return 0; // null string
      
      string_view d_str = get_column().template element<string_view>(idx);
      char* d_buffer = get_chars() + get_offsets()[idx];
      
      for( auto itr = d_str.begin(); itr != d_str.end(); ++itr ) {
        auto the_chr = *itr;
        uint32_t code_point = detail::utf8_to_codepoint(the_chr);
        detail::character_flags_table_type flag = code_point <= 0x00FFFF ? get_flags()[code_point] : 0;

        if( (itr == d_str.begin()) ? IS_LOWER(flag) : IS_UPPER(flag) )
          the_chr = detail::codepoint_to_utf8(get_case_table()[code_point]);
        d_buffer += detail::from_char_utf8(the_chr, d_buffer);
      }
      return 0;
    }
  };


  //functor for probing string title-ization
  //requirements:
  //
  struct probe_title: probe_base
  {
    probe_title(column_device_view const d_column,
                     character_flags_table_type const* d_flags,
                     character_cases_table_type const* d_case_table):
      probe_base(d_column, d_flags, d_case_table)
    {  
    }
    
     __device__
     int32_t operator()(size_type idx) const override {
       if( get_column().is_null(idx) )
         return 0; // null string
      
       string_view d_str = get_column().template element<string_view>(idx);
       int32_t bytes = 0;

       bool bcapnext = true;
       for( auto itr = d_str.begin(); itr != d_str.end(); ++itr ) {
         auto the_chr = *itr;
         uint32_t code_point = detail::utf8_to_codepoint(the_chr);
         detail::character_flags_table_type flag = code_point <= 0x00FFFF ? get_flags()[code_point] : 0;

         if( !IS_ALPHA(flag) ) {
           bcapnext = true;
           bytes += detail::bytes_in_char_utf8(the_chr);
         }
         else if( (bcapnext && IS_LOWER(flag)) || (!bcapnext && IS_UPPER(flag)) ) {
           bcapnext = false;
           bytes += detail::bytes_in_char_utf8(detail::codepoint_to_utf8(get_case_table()[code_point]));
         }
         else
           bytes += detail::bytes_in_char_utf8(the_chr);
       }
       return bytes;
    }
  };

  //functor for executing string title-ization:
  //
  struct execute_title: execute_base
  {
    execute_title(column_device_view const d_column,
                  character_flags_table_type const* d_flags,
                  character_cases_table_type const* d_case_table,
                  int32_t const* d_offsets,
                  char* d_chars):
      execute_base(d_column, d_flags, d_case_table, d_offsets, d_chars)
    {
    }
    
    __device__
    int32_t operator()(size_type idx) override {
      if( get_column().is_null(idx) )
        return 0; // null string
      
      string_view d_str = get_column().template element<string_view>(idx);
      char* d_buffer = get_chars() + get_offsets()[idx];

      bool bcapnext = true;
      for( auto itr = d_str.begin(); itr != d_str.end(); ++itr ) {
        auto the_chr = *itr;
        uint32_t code_point = detail::utf8_to_codepoint(the_chr);
        detail::character_flags_table_type flag = code_point <= 0x00FFFF ? get_flags()[code_point] : 0;

        if( !IS_ALPHA(flag) )
          bcapnext = true;
        else
          {
            if( bcapnext ? IS_LOWER(flag) : IS_UPPER(flag) )
              the_chr = detail::codepoint_to_utf8(get_case_table()[code_point]);
            bcapnext = false;
          }
        
        d_buffer += detail::from_char_utf8(the_chr, d_buffer);
      }
      return 0;
    }
  };

         
}//anonym.

template<typename device_probe_functor,
         typename device_execute_functor>
std::unique_ptr<column> modify_strings( strings_column_view const& strings,
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

  device_probe_functor d_probe_fctr{d_column, d_flags, d_case_table};

  // build offsets column -- calculate the size of each output string
  auto offsets_transformer_itr = thrust::make_transform_iterator( thrust::make_counting_iterator<size_type>(0), d_probe_fctr);
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

  device_execute_functor d_execute_fctr{d_column,
      d_flags,
      d_case_table,
      d_new_offsets,
      d_chars};
  
  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<size_type>(0), strings_count, d_execute_fctr);
  
  //
  return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                             d_column.null_count(), std::move(null_mask), stream, mr);
}

}//namespace detail

std::unique_ptr<column> capitalize( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  return detail::modify_strings<detail::probe_capitalize, detail::execute_capitalize>(strings, mr);
}

std::unique_ptr<column> title( strings_column_view const& strings,
                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
{
  return detail::modify_strings<detail::probe_title, detail::execute_title>(strings, mr);
}
  
}//namespace strings
}//namespace cudf
