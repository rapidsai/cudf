/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/strings/udf/case.cuh>
#include <cudf/strings/udf/char_types.cuh>
#include <cudf/strings/udf/search.cuh>
#include <cudf/strings/udf/starts_with.cuh>
#include <cudf/strings/udf/strip.cuh>
#include <cudf/strings/udf/udf_string.cuh>

using namespace cudf::strings::udf;

extern "C" __device__ int len(int* nb_retval, void const* str)
{
  auto sv    = reinterpret_cast<cudf::string_view const*>(str);
  *nb_retval = sv->length();
  return 0;
}

extern "C" __device__ int startswith(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = starts_with(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int endswith(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = ends_with(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int contains(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = (str_view->find(*substr_view) != cudf::string_view::npos);
  return 0;
}

extern "C" __device__ int find(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = str_view->find(*substr_view);
  return 0;
}

extern "C" __device__ int rfind(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = str_view->rfind(*substr_view);
  return 0;
}

extern "C" __device__ int eq(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view == *rhs_view);
  return 0;
}

extern "C" __device__ int ne(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view != *rhs_view);
  return 0;
}

extern "C" __device__ int ge(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view >= *rhs_view);
  return 0;
}

extern "C" __device__ int le(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view <= *rhs_view);
  return 0;
}

extern "C" __device__ int gt(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view > *rhs_view);
  return 0;
}

extern "C" __device__ int lt(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view < *rhs_view);
  return 0;
}

extern "C" __device__ int pyislower(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_lower(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisupper(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_upper(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisspace(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_space(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisdecimal(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_decimal(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisnumeric(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_numeric(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisdigit(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_digit(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisalnum(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_alpha_numeric(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisalpha(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_alpha(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyistitle(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_title(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pycount(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = count(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int udf_string_from_string_view(int* nb_retbal,
                                                      void const* str,
                                                      void* udf_str)
{
  auto str_view_ptr = reinterpret_cast<cudf::string_view const*>(str);
  auto udf_str_ptr  = new (udf_str) udf_string;
  *udf_str_ptr      = udf_string(*str_view_ptr);

  return 0;
}

extern "C" __device__ int strip(int* nb_retval,
                                void* udf_str,
                                void* const* to_strip,
                                void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr   = new (udf_str) udf_string;

  *udf_str_ptr = strip(*to_strip_ptr, *strip_str_ptr);

  return 0;
}

extern "C" __device__ int lstrip(int* nb_retval,
                                 void* udf_str,
                                 void* const* to_strip,
                                 void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr   = new (udf_str) udf_string;

  *udf_str_ptr = strip(*to_strip_ptr, *strip_str_ptr, cudf::strings::side_type::LEFT);

  return 0;
}

extern "C" __device__ int rstrip(int* nb_retval,
                                 void* udf_str,
                                 void* const* to_strip,
                                 void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr   = new (udf_str) udf_string;

  *udf_str_ptr = strip(*to_strip_ptr, *strip_str_ptr, cudf::strings::side_type::RIGHT);

  return 0;
}
extern "C" __device__ int upper(int* nb_retval,
                                void* udf_str,
                                void const* st,
                                std::uintptr_t flags_table,
                                std::uintptr_t cases_table,
                                std::uintptr_t special_table)
{
  auto udf_str_ptr = new (udf_str) udf_string;
  auto st_ptr      = reinterpret_cast<cudf::string_view const*>(st);

  auto flags_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(flags_table);
  auto cases_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_cases_table_type*>(cases_table);
  auto special_table_ptr =
    reinterpret_cast<cudf::strings::detail::special_case_mapping*>(special_table);

  cudf::strings::udf::chars_tables tables{flags_table_ptr, cases_table_ptr, special_table_ptr};

  *udf_str_ptr = to_upper(tables, *st_ptr);

  return 0;
}

extern "C" __device__ int lower(int* nb_retval,
                                void* udf_str,
                                void const* st,
                                std::uintptr_t flags_table,
                                std::uintptr_t cases_table,
                                std::uintptr_t special_table)
{
  auto udf_str_ptr = new (udf_str) udf_string;
  auto st_ptr      = reinterpret_cast<cudf::string_view const*>(st);

  auto flags_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(flags_table);
  auto cases_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_cases_table_type*>(cases_table);
  auto special_table_ptr =
    reinterpret_cast<cudf::strings::detail::special_case_mapping*>(special_table);

  cudf::strings::udf::chars_tables tables{flags_table_ptr, cases_table_ptr, special_table_ptr};
  *udf_str_ptr = to_lower(tables, *st_ptr);
  return 0;
}

extern "C" __device__ int concat(int* nb_retval, void* udf_str, void* const* lhs, void* const* rhs)
{
  auto lhs_ptr = reinterpret_cast<cudf::string_view const*>(lhs);
  auto rhs_ptr = reinterpret_cast<cudf::string_view const*>(rhs);

  auto udf_str_ptr = new (udf_str) udf_string;

  udf_string result;
  result.append(*lhs_ptr).append(*rhs_ptr);
  *udf_str_ptr = result;
  return 0;
}
