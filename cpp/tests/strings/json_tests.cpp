/*
 * Copyright (c) 2021, BAIDU CORPORATION.
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
#include <cudf/strings/json.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

struct JsonTests : public cudf::test::BaseFixture {
};

TEST_F(JsonTests, jsonObjectToArray)
{
  {
    /*
    const char* str3 = "{\n\t\t\t\"category\": \"reference\",\n\t\t\t\"author\": "
    "\"Nigel Rees\",\n\t\t\t\"title\": \"Sayings of the Century\",\n\t\t\t\"price\": "
    "8.95\n\t\t}";
    */
    std::string json_string {
        "{"
           "\"category\": \"reference\","
           "\"author\": \"Nigel Rees\","
           "\"title\": \"Sayings of the Century\","
           "\"price\": 8.95"
        "}"
    };

    cudf::test::strings_column_wrapper input{json_string};
    auto result = cudf::strings::json_to_array(cudf::strings_column_view(input));

    cudf::test::strings_column_wrapper expected_key_column({"category", "author", "title", "price"});
    cudf::test::strings_column_wrapper expected_value_column({"reference", "Nigel Rees", "Sayings of the Century", "8.95"});
    auto struct_col = 
      cudf::test::structs_column_wrapper({expected_key_column, expected_value_column}, {1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col = cudf::column(*struct_col);
    cudf::test::expect_columns_equivalent(expected_unchanged_struct_col,
                                          cudf::lists_column_view(*result).child());
  }
}

TEST_F(JsonTests, JsonNestObjectToArray)
{
  {
    std::string json_string {
      "{"
        "\"store\": {\"book\": ["
            "{"
              "\"category\": \"reference\","
              "\"author\": \"Nigel Rees\","
              "\"title\": \"Sayings of the Century\","
              "\"price\": 8.95"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"Evelyn Waugh\","
              "\"title\": \"Sword of Honour\","
              "\"price\": 12.99"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"Herman Melville\","
              "\"title\": \"Moby Dick\","
              "\"isbn\": \"0-553-21311-3\","
              "\"price\": 8.99"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"J. R. R. Tolkien\","
              "\"title\": \"The Lord of the Rings\","
              "\"isbn\": \"0-395-19395-8\","
              "\"price\": 22.99"
            "}"
          "],"
          "\"bicycle\": {"
            "\"color\": \"red\","
            "\"price\": 19.95"
          "}"
        "},"
        "\"expensive\": 10"
      "}"
    };
    cudf::test::strings_column_wrapper input{json_string};
    auto result = cudf::strings::json_to_array(cudf::strings_column_view(input));

    cudf::test::strings_column_wrapper expected_key_column({"store", "expensive"});
    std::string v1 {
        "{\"book\": ["
            "{"
              "\"category\": \"reference\","
              "\"author\": \"Nigel Rees\","
              "\"title\": \"Sayings of the Century\","
              "\"price\": 8.95"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"Evelyn Waugh\","
              "\"title\": \"Sword of Honour\","
              "\"price\": 12.99"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"Herman Melville\","
              "\"title\": \"Moby Dick\","
              "\"isbn\": \"0-553-21311-3\","
              "\"price\": 8.99"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"J. R. R. Tolkien\","
              "\"title\": \"The Lord of the Rings\","
              "\"isbn\": \"0-395-19395-8\","
              "\"price\": 22.99"
            "}"
          "],"
          "\"bicycle\": {"
            "\"color\": \"red\","
            "\"price\": 19.95"
          "}"
        "}"
    };
    cudf::test::strings_column_wrapper expected_value_column({v1, "10"});
    auto struct_col = 
      cudf::test::structs_column_wrapper({expected_key_column, expected_value_column}, {1, 1}).release();
    auto expected_unchanged_struct_col = cudf::column(*struct_col);
    cudf::test::expect_columns_equivalent(expected_unchanged_struct_col,
                                          cudf::lists_column_view(*result).child());
  }

  {
    //const char* str2 = "{\n\"123\": 132124,\n\"vasdfsss\": [\n1,\n2,\n\"fdsfsd\",\n{\n\"dsfdf\": 244\n}\n],\n\"fsfsd\": 5245\n }\n";
  
    std::string json_string {
        "{"
           "\"123\": 132124,"
           "\"vasdfsss\": ["
              "1,"
              "2,"
              "\"fdsfsd\","
              "{"
                  "\"dsfdf\": 244"
              "}"
            "],"
            "\"fsfsd\": 5245"
        "}"
    };
    cudf::test::strings_column_wrapper input{json_string};
    auto result = cudf::strings::json_to_array(cudf::strings_column_view(input));

    cudf::test::strings_column_wrapper expected_key_column({"123", "vasdfsss", "fsfsd"});
    std::string v2 = {
      "["
        "1,"
        "2,"
        "\"fdsfsd\","
        "{"
            "\"dsfdf\": 244"
        "}"
      "]"
    };
    cudf::test::strings_column_wrapper expected_value_column({"132124", v2, "5245"});
    auto struct_col = 
      cudf::test::structs_column_wrapper({expected_key_column, expected_value_column}, {1, 1, 1}).release();
    auto expected_unchanged_struct_col = cudf::column(*struct_col);
    cudf::test::expect_columns_equivalent(expected_unchanged_struct_col,
                                          cudf::lists_column_view(*result).child());
  }

  {
    /*
    const char* medc_str2 = "{\"logid\":\"2801611157\",\"service\":{\"aps\":{\"count\":\"1,1,0\","
    "\"items\":[{\"product\":\"11/box.rnplugin.feedhn\",\"valid\":\"1\",\"version\":1605497738}]}},"
    "\"traceid\":\"c1140a6cb60a4da3b66572e0e8f08e8b\"}";
    */
    std::string json_string {
      "{"
          "\"logid\":\"2801611157\","
          "\"service\":{"
                "\"aps\":{"
                    "\"count\":1,1,0\","
                    "\"items\":["
                        "{"
                            "\"product\":\"11/box.rnplugin.feedhn\","
                            "\"valid\":\"1\","
                            "\"version\":1605497738"
                        "}"
                    "]"
                "}"
          "},"
          "\"traceid\":\"c1140a6cb60a4da3b66572e0e8f08e8b\""
      "}"
    };
    cudf::test::strings_column_wrapper input{json_string};
    auto result = cudf::strings::json_to_array(cudf::strings_column_view(input));

    cudf::test::strings_column_wrapper expected_key_column({"logid", "service", "traceid"});
    cudf::test::strings_column_wrapper expected_value_column({"2801611157", "{"
                "\"aps\":{"
                    "\"count\":1,1,0\","
                    "\"items\":["
                        "{"
                            "\"product\":\"11/box.rnplugin.feedhn\","
                            "\"valid\":\"1\","
                            "\"version\":1605497738"
                        "}"
                    "]"
                "}"
          "}", "c1140a6cb60a4da3b66572e0e8f08e8b"});
    auto struct_col = 
      cudf::test::structs_column_wrapper({expected_key_column, expected_value_column}, {1, 1, 1}).release();
    auto expected_unchanged_struct_col = cudf::column(*struct_col);
    cudf::test::expect_columns_equivalent(expected_unchanged_struct_col,
                                          cudf::lists_column_view(*result).child());
  }
}

TEST_F(JsonTests, JsonArrayToArray)
{
  /*
  const char* str1 = "[\n\t\t{\n\t\t\t\"category\": \"reference\",\n\t\t\t\"author\": "
  "\"Nigel Rees\",\n\t\t\t\"title\": \"Sayings of the Century\",\n\t\t\t\"price\": "
  "8.95\n\t\t},\n\t\t{\n\t\t\t\"category\": \"fiction\",\n\t\t\t\"author\": \"Evelyn "
  "Waugh\",\n\t\t\t\"title\": \"Sword of Honour\",\n\t\t\t\"price\": "
  "12.99\n\t\t}]";*/
  {
    std::string json_string {
      "["
          "{"
              "\"category\": \"reference\","
              "\"author\": \"Nigel Rees\","
              "\"title\": \"Sayings of the Century\","
              "\"price\": 8.95"
          "},"
          "{"
              "\"category\": \"fiction\","
              "\"author\": \"Evelyn Waugh\","
              "\"title\": \"Sword of Honour\","
              "\"price\": 12.99"
          "}"
      "]"
    };
    cudf::test::strings_column_wrapper input{json_string};
    auto result = cudf::strings::json_to_array(cudf::strings_column_view(input));

    cudf::test::strings_column_wrapper expected_key_column({"category", "author", "title", "price", 
                                                            "category", "author", "title", "price"});
    cudf::test::strings_column_wrapper expected_value_column({"reference", "Nigel Rees", "Sayings of the Century", "8.95",
                                                              "fiction", "Evelyn Waugh", "Sword of Honour", "12.99"});
    auto struct_col = 
      cudf::test::structs_column_wrapper({expected_key_column, expected_value_column}, {1, 1, 1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col = cudf::column(*struct_col);
    cudf::test::expect_columns_equivalent(expected_unchanged_struct_col,
                                          cudf::lists_column_view(*result).child());
  }
}

TEST_F(JsonTests, JsonToArrayContainsUrl)
{
  {
    /*
    const char* medc_str1 = "{\"pd\":\"feed\",\"mt\":2,\"searchID\":\"58b7e0bba66f703a\",\"vType\":0,\"tab\":\"1\","
    "\"locid\":\"http://www.internal.video.baidu.com/a3e3f0d392b4885171c9fc179662bde2.html\","
    "\"duration\":11,\"authorID\":\"1679702981602955\",\"mtNew\":\"na\",\"pdRec\":\"feed\","
    "\"refresh_timestamp_ms\":1605631869224,\"oper_type\":\"up_down\",\"vid\":\"4834619723738471155\","
    "\"refreshTimestampMs\":1605631869224,\"step\":\"1\",\"nid_src\":\"4834619723738471155\",\"ext_page\":\"mini_video_landing\","
    "\"clickID\":\"93e98109624f3f562d4a473f195fa455\","
    "\"url\":\"http://vd4.bdstatic.com/mda-kkcwwxka99rv9jmt/v1-cae/1080p/mda-kkcwwxka99rv9jmt.mp4?v_from_s=bdapp-unicomment-shunyi"
    "&vt=0&cd=0&did=b809fa572adbfd45f145caafd9ad7709&logid=3069223481&vid=4834619723738471155&pd=0&pt=0&av=12.3.0.11&cr=3&sle=1&sl="
    "1053&split=1078656&auth_key=1605639069-0-0-8fd52dee97e99da61a9afc8b5fcb0445&bcevod_channel=searchbox_feed&pdx=0&nt=0&dt=1\","
    "\"netType\":\"wifi\",\"currentPosition\":0,\"auto_play\":0}";
    */
    std::string json_string {
      "{"
          "\"pd\":\"feed\","
          "\"mt\":2,"
          "\"searchID\":\"58b7e0bba66f703a\","
          "\"locid\":\"http://www.internal.video.baidu.com/a3e3f0d392b4885171c9fc179662bde2.html\","
          "\"authorID\":\"1679702981602955\","
          "\"url\":\"http://vd4.bdstatic.com/mda-kkcwwxka99rv9jmt/v1-cae/1080p/mda-kkcwwxka99rv9jmt.mp4?v_from_s="
          "bdapp-unicomment-shunyi&vt=0&cd=0&did=b809fa572adbfd45f145caafd9ad7709&logid=3069223481&vid=48346197237"
          "38471155&pd=0&pt=0&av=12.3.0.11&cr=3&sle=1&sl=1053&split=1078656&auth_key=1605639069-0-0-8fd52dee97e99d"
          "a61a9afc8b5fcb0445&bcevod_channel=searchbox_feed&pdx=0&nt=0&dt=1\","
          "\"currentPosition\":0,"
          "\"auto_play\":0"
      "}"
    };
    cudf::test::strings_column_wrapper input{json_string};
    auto result = cudf::strings::json_to_array(cudf::strings_column_view(input));

    cudf::test::strings_column_wrapper expected_key_column({"pd", "mt", "searchID", "locid", 
                                                            "authorID", "url", "currentPosition", "auto_play"});
    std::string url1 {"http://www.internal.video.baidu.com/a3e3f0d392b4885171c9fc179662bde2.html"};
    std::string url2 {"http://vd4.bdstatic.com/mda-kkcwwxka99rv9jmt/v1-cae/1080p/mda-kkcwwxka99rv9jmt.mp4?v_from_s="
          "bdapp-unicomment-shunyi&vt=0&cd=0&did=b809fa572adbfd45f145caafd9ad7709&logid=3069223481&vid=48346197237"
          "38471155&pd=0&pt=0&av=12.3.0.11&cr=3&sle=1&sl=1053&split=1078656&auth_key=1605639069-0-0-8fd52dee97e99d"
          "a61a9afc8b5fcb0445&bcevod_channel=searchbox_feed&pdx=0&nt=0&dt=1"};
    cudf::test::strings_column_wrapper expected_value_column({"feed", "2", "58b7e0bba66f703a", url1,
                                                              "1679702981602955", url2, "0", "0"});
    auto struct_col = 
      cudf::test::structs_column_wrapper({expected_key_column, expected_value_column}, {1, 1, 1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col = cudf::column(*struct_col);
    cudf::test::expect_columns_equivalent(expected_unchanged_struct_col,
                                          cudf::lists_column_view(*result).child());
  }
}

CUDF_TEST_PROGRAM_MAIN()
