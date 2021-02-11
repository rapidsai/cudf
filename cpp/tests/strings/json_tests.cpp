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

/*
const char* json_string = "{
    "store": {
        "book": [
            {
                "category": "reference",
                "author": "Nigel Rees",
                "title": "Sayings of the Century",
                "price": 8.95
            },
            {
                "category": "fiction",
                "author": "Evelyn Waugh",
                "title": "Sword of Honour",
                "price": 12.99
            },
            {
                "category": "fiction",
                "author": "Herman Melville",
                "title": "Moby Dick",
                "isbn": "0-553-21311-3",
                "price": 8.99
            },
            {
                "category": "fiction",
                "author": "J. R. R. Tolkien",
                "title": "The Lord of the Rings",
                "isbn": "0-395-19395-8",
                "price": 22.99
            }
        ],
        "bicycle": {
            "color": "red",
            "price": 19.95
        }
    },
}";
*/

struct JsonTests : public cudf::test::BaseFixture {
};

TEST_F(JsonTests, GetJsonObject)
{
  // reference:  https://jsonpath.herokuapp.com/
  // clang-format off
   /*
   {
      "store": {
         "book": [
               {
                  "category": "reference",
                  "author": "Nigel Rees",
                  "title": "Sayings of the Century",
                  "price": 8.95
               },
               {
                  "category": "fiction",
                  "author": "Evelyn Waugh",
                  "title": "Sword of Honour",
                  "price": 12.99
               },
               {
                  "category": "fiction",
                  "author": "Herman Melville",
                  "title": "Moby Dick",
                  "isbn": "0-553-21311-3",
                  "price": 8.99
               },
               {
                  "category": "fiction",
                  "author": "J. R. R. Tolkien",
                  "title": "The Lord of the Rings",
                  "isbn": "0-395-19395-8",
                  "price": 22.99
               }
         ],
         "bicycle": {
               "color": "red",
               "price": 19.95
         }
      },
      "expensive": 10
   }
   */
  // clang-format on
  // this string is formatted to result in a reasonably readable debug printf
  const char* json_string =
    "{\n\"store\": {\n\t\"book\": [\n\t\t{\n\t\t\t\"category\": \"reference\",\n\t\t\t\"author\": "
    "\"Nigel Rees\",\n\t\t\t\"title\": \"Sayings of the Century\",\n\t\t\t\"price\": "
    "8.95\n\t\t},\n\t\t{\n\t\t\t\"category\": \"fiction\",\n\t\t\t\"author\": \"Evelyn "
    "Waugh\",\n\t\t\t\"title\": \"Sword of Honour\",\n\t\t\t\"price\": "
    "12.99\n\t\t},\n\t\t{\n\t\t\t\"category\": \"fiction\",\n\t\t\t\"author\": \"Herman "
    "Melville\",\n\t\t\t\"title\": \"Moby Dick\",\n\t\t\t\"isbn\": "
    "\"0-553-21311-3\",\n\t\t\t\"price\": 8.99\n\t\t},\n\t\t{\n\t\t\t\"category\": "
    "\"fiction\",\n\t\t\t\"author\": \"J. R. R. Tolkien\",\n\t\t\t\"title\": \"The Lord of the "
    "Rings\",\n\t\t\t\"isbn\": \"0-395-19395-8\",\n\t\t\t\"price\": "
    "22.99\n\t\t}\n\t],\n\t\"bicycle\": {\n\t\t\"color\": \"red\",\n\t\t\"price\": "
    "19.95\n\t}\n},\n\"expensive\": 10\n}";

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store.book");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store.*");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store.book[*]");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store.book[*].category");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store.book[*].title");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store['bicycle']");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store.book[*]['isbn']");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string json_path("$.store.book[2]");
    auto result = cudf::strings::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::print(*result);
  }
}
