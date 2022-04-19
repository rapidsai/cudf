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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/udf/column_functions.hpp>
#include <cudf/strings/udf/dstring.cuh>
#include <cudf/utilities/span.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

struct DStringTest : public cudf::test::BaseFixture {
};

namespace {

template <typename Functor>
void run_dstring_test(Functor fn, cudf::column_view const expected)
{
  auto const rows  = cudf::column_view(expected).size();
  auto output      = cudf::strings::udf::create_dstring_array(rows);
  auto output_data = static_cast<cudf::strings::udf::dstring*>(output->data());
  auto stream      = rmm::cuda_stream_default;
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(rows),
                    output_data,
                    fn);

  auto d_span  = cudf::device_span<cudf::strings::udf::dstring const>(output_data, rows);
  auto results = cudf::strings::udf::make_strings_column(d_span);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  cudf::strings::udf::free_dstring_array(
    cudf::device_span<cudf::strings::udf::dstring>(output_data, rows));
}

struct ctor_and_assign_fn {
  using dstring = cudf::strings::udf::dstring;
  __device__ dstring operator()(int idx)
  {
    switch (idx) {
      case 0: {  // dstring()
        dstring result;
        return result;
      }
      case 1: {  // dstring(char*)
        dstring result("hello");
        return result;
      }
      case 2: {  // dstring(char*,int)
        dstring result("goodbye", 4);
        return result;
      }
      case 3: {  // dstring(string_view)
        cudf::string_view sv("world", 5);
        dstring result(sv);
        return result;
      }
      case 4: {  // dstring(int,char)
        dstring result(5, '#');
        return result;
      }
      case 5: {  // dstring(dstring&)
        dstring input("copy");
        dstring result(input);
        return result;
      }
      case 6: {  // dstring(dstring&&)
        dstring input("move");
        dstring result(std::move(input));
        return result;
      }
      case 7: {  // operator=(char*)
        dstring result;
        result = "hello";
        return result;
      }
      case 8: {  // operator=(string_view)
        cudf::string_view sv("world", 5);
        dstring result;
        result = sv;
        return result;
      }
      case 9: {  // operator=(dstring&)
        dstring result, input("copied");
        result = input;
        return result;
      }
      case 10: {  // operator=(dstring&&)
        dstring result;
        result = dstring("moved");
        return result;
      }
      case 11: {  // operator=(char*)
        dstring result;
        result = "hello";
        return result;
      }
      case 12: {  // assign(char*)
        dstring result;
        return result.assign("accénted");
      }
      case 13: {  // assign(char*,int)
        dstring result;
        return result.assign("goodbye", 4);
      }
      case 14: {  // assign(string_view)
        cudf::string_view sv("world", 5);
        dstring result;
        return result.assign(sv);
      }
      case 15: {  // operator=(dstring&&)
        dstring result;
        return result.assign(dstring("movié"));
      }
    }
  }
};

}  // namespace

TEST_F(DStringTest, Constructors)
{
  cudf::test::strings_column_wrapper expected({"",
                                               "hello",
                                               "good",
                                               "world",
                                               "#####",
                                               "copy",
                                               "move",
                                               "hello",
                                               "world",
                                               "copied",
                                               "moved",
                                               "hello",
                                               "accénted",
                                               "good",
                                               "world",
                                               "movié"},
                                              {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  run_dstring_test(ctor_and_assign_fn{}, expected);
}

namespace {

struct append_fn {
  using dstring = cudf::strings::udf::dstring;
  __device__ dstring operator()(int idx)
  {
    dstring result(":::");
    cudf::string_view sv("world", 5);
    switch (idx) {
      case 0: return result.append("hello");       // append(char*)
      case 1: return result.append("goodbye", 4);  // append(char*,int)
      case 2: return result.append(sv);            // append(string_view)
      case 3: return result.append('$', 6);        // append(char,int)
      case 4: return result += "accénted";         // operator+=(char*)
      case 5: return result += sv;                 // operator+=(string_view)
      case 6: return result += '?';                // operator+=(char)
    }
  }
};

}  // namespace

TEST_F(DStringTest, Append)
{
  cudf::test::strings_column_wrapper expected(
    {":::hello", ":::good", ":::world", ":::$$$$$$", ":::accénted", ":::world", ":::?"});
  run_dstring_test(append_fn{}, expected);
}

namespace {

struct insert_fn {
  using dstring = cudf::strings::udf::dstring;
  __device__ dstring operator()(int idx)
  {
    dstring result(":::::");
    cudf::string_view sv("insert", 6);
    switch (idx) {
      case 0: return result.insert(1, "héllo");        // insert(char*)
      case 1: return result.insert(2, "good day", 4);  // insert(char*,int)
      case 2: return result.insert(3, sv);             // insert(string_view)
      case 3: return result.insert(4, 2, '_');         // insert(char,int)
      case 4: return result.insert(0, "héllo");
      case 5: return result.insert(result.length(), "héllo");
      case 6: return result.insert(-1, "héllo");
      case 7: return result.insert(result.length(), 1, 'X');
    }
  }
};

}  // namespace

TEST_F(DStringTest, Insert)
{
  cudf::test::strings_column_wrapper expected({":héllo::::",
                                               "::good:::",
                                               ":::insert::",
                                               "::::__:",
                                               "héllo:::::",
                                               ":::::héllo",
                                               ":::::",
                                               ":::::X"});

  run_dstring_test(insert_fn{}, expected);
}

namespace {

struct replace_fn {
  using dstring = cudf::strings::udf::dstring;
  __device__ dstring operator()(int idx)
  {
    dstring result("0123456789");
    cudf::string_view sv("replace", 7);
    switch (idx) {
      case 0: return result.replace(5, 1, " ");                  // replace(char*): same size
      case 1: return result.replace(3, 2, "XYZ", 2);             // replace(char*,int): same size
      case 2: return result.replace(2, 7, sv);                   // replace(string_view): same size
      case 3: return result.replace(1, 2, 2, '_');               // replace(char,int): same size
      case 4: return result.replace(2, 5, "XY");                 // replace(char*): smaller
      case 5: return result.replace(0, 6, "XYZ", 2);             // replace(char*,int): smaller
      case 6: return result.replace(1, 8, sv);                   // replace(string_view): smaller
      case 7: return result.replace(3, 4, 2, '_');               // replace(char,int): smaller
      case 8: return result.replace(2, 2, "WXYZ");               // replace(char*): larger
      case 9: return result.replace(0, 2, "WXYZ", 3);            // replace(char*,int): larger
      case 10: return result.replace(1, 4, sv);                  // replace(string_view): larger
      case 11: return result.replace(3, 4, 6, '_');              // replace(char,int): larger
      case 12: return result.replace(5, -1, "");                 // replace to the end of a string
      case 13: return result.replace(5, 0, " ");                 // replace/insert
      case 14: return result.replace(result.length(), -1, "X");  // replace/append
      case 15: return result.replace(-1, -1, "X");               // no change
      case 16: return result.replace(0, -1, "X", -1);            // no change
      case 17: return result.replace(3, 0, "");                  // no change
    }
  }
};

}  // namespace

TEST_F(DStringTest, Replace)
{
  cudf::test::strings_column_wrapper expected({
    "01234 6789",     // 0
    "012XY56789",     // 1
    "01replace9",     // 2
    "0__3456789",     // 3
    "01XY789",        // 4
    "XY6789",         // 5
    "0replace9",      // 6
    "012__789",       // 7
    "01WXYZ456789",   // 8
    "WXY23456789",    // 9
    "0replace56789",  // 10
    "012______789",   // 11
    "01234",          // 12
    "01234 56789",    // 13
    "0123456789X",    // 14
    "0123456789",     // 15
    "0123456789",     // 16
    "0123456789"      // 17
  });

  run_dstring_test(replace_fn{}, expected);
}

namespace {

struct erase_fn {
  using dstring = cudf::strings::udf::dstring;
  __device__ dstring operator()(int idx)
  {
    dstring result("0123456789");
    switch (idx) {
      case 0: return result.erase(5, 5);
      case 1: return result.erase(5);
      case 2: return result.erase(0, 5);
      case 3: return result.erase(0);
      case 4: return result.erase(-1);
      case 5: return result.erase(result.length());
    }
  }
};

}  // namespace

TEST_F(DStringTest, Erase)
{
  cudf::test::strings_column_wrapper expected(
    {"01234", "01234", "56789", "", "0123456789", "0123456789"});
  run_dstring_test(erase_fn{}, expected);
}

namespace {

struct substring_fn {
  using dstring = cudf::strings::udf::dstring;
  __device__ dstring operator()(int idx)
  {
    dstring result("0123456789");
    switch (idx) {
      case 0: return result.substr(0, 5);
      case 1: return result.substr(5);
      case 2: return result.substr(5, result.length());
      case 3: return result.substr(0);
      case 4: return result.substr(-1);
      case 5: return result.substr(0, result.length());
      case 6: return result.substr(3, 0);
      case 7: return result.substr(result.length(), 0);
    }
  }
};

}  // namespace

TEST_F(DStringTest, Substring)
{
  cudf::test::strings_column_wrapper expected(
    {"01234", "56789", "56789", "0123456789", "", "0123456789", "", ""});
  run_dstring_test(substring_fn{}, expected);
}

namespace {

struct resize_fn {
  using dstring = cudf::strings::udf::dstring;
  __device__ cudf::size_type operator()(int idx)
  {
    dstring result("0123456789");
    switch (idx) {
      case 0: result.reserve(5); break;
      case 1: result.reserve(25); break;
      case 2: result.resize(12); break;
      case 3: result.resize(4); break;
      case 4: result.clear(); break;
      case 5:
        result.reserve(25);
        result.shrink_to_fit();
        break;
      case 6:
        result.resize(12);
        result.shrink_to_fit();
        break;
    }
    return result.size_bytes();
  }
};

}  // namespace

TEST_F(DStringTest, Resize)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected({10, 10, 12, 4, 0, 10, 12});

  auto rows     = cudf::column_view(expected).size();
  auto stream   = rmm::cuda_stream_default;
  auto d_result = rmm::device_uvector<cudf::size_type>(rows, stream);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(rows),
                    d_result.data(),
                    resize_fn{});

  auto d_span = cudf::device_span<cudf::size_type const>(d_result.data(), rows);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(cudf::column_view(d_span), expected);
}

namespace {

struct compare_fn {
  using dstring = cudf::strings::udf::dstring;
  __device__ int operator()(cudf::string_view const sv)
  {
    dstring result("abcdef");
    auto const rtn = result.compare(sv);
    // convert to simply 0, 1, or -1
    return rtn == 0 ? 0 : (rtn / std::abs(rtn));
  }
};

}  // namespace

TEST_F(DStringTest, Compare)
{
  cudf::test::strings_column_wrapper input({"abcdef", "abcdefg", "abcdéf", "012345", "abc", ""});
  auto rows   = cudf::column_view(input).size();
  auto stream = rmm::cuda_stream_default;

  auto view_array = cudf::strings::udf::create_string_view_array(cudf::strings_column_view(input));
  auto d_result   = rmm::device_uvector<int>(rows, stream);

  thrust::transform(
    rmm::exec_policy(stream), view_array.begin(), view_array.end(), d_result.data(), compare_fn{});

  auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, -1, -1, 1, 1, 1});
  auto d_span   = cudf::device_span<cudf::size_type const>(d_result.data(), rows);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(cudf::column_view(d_span), expected);
}

namespace {

struct copy_fn {
  cudf::string_view const* d_input;

  using dstring = cudf::strings::udf::dstring;
  __device__ dstring operator()(int idx)
  {
    dstring result{d_input[idx]};
    return result;
  }
};

}  // namespace

TEST_F(DStringTest, ColumnFunctions)
{
  cudf::test::strings_column_wrapper input({"abcdef", "abcdefg", "abcdéf", "012345", "abc", ""});
  auto view_array = cudf::strings::udf::create_string_view_array(cudf::strings_column_view(input));

  run_dstring_test(copy_fn{view_array.data()}, input);
}
