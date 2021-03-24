/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/substring.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

struct JsonTests : public cudf::test::BaseFixture {
};

// json array
  const char* str1 = "[\n\t\t{\n\t\t\t\"category\": \"reference\",\n\t\t\t\"author\": "
    "\"Nigel Rees\",\n\t\t\t\"title\": \"Sayings of the Century\",\n\t\t\t\"price\": "
    "8.95\n\t\t},\n\t\t{\n\t\t\t\"category\": \"fiction\",\n\t\t\t\"author\": \"Evelyn "
    "Waugh\",\n\t\t\t\"title\": \"Sword of Honour\",\n\t\t\t\"price\": "
    "12.99\n\t\t}]";
  // json obejct
  const char* str2 = "{\n\"123\": 132124,\n\"vasdfsss\": [\n1,\n2,\n\"fdsfsd\",\n{\n\"dsfdf\": 244\n}\n],\n\"fsfsd\": 5245\n }\n";
  // json object
  const char* str3 = "{\n\t\t\t\"category\": \"reference\",\n\t\t\t\"author\": "
    "\"Nigel Rees\",\n\t\t\t\"title\": \"Sayings of the Century\",\n\t\t\t\"price\": "
    "8.95\n\t\t}";
  {
    cudf::test::strings_column_wrapper input{str1};
    auto result = cudf::strings::json_to_array(cudf::strings_column_view(input));
    cudf::test::print(*result);
  }
  
  {
    cudf::test::strings_column_wrapper input{str};
    auto result = cudf::strings::json_to_array(cudf::strings_column_view(input));
    cudf::test::print(*result);
  }
