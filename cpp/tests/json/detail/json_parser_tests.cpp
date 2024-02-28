/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>

#include <cudf/json/detail/json_parser.hpp>
#include <cudf/json/json.hpp>

struct JsonParserTests : public cudf::test::BaseFixture {};
using cudf::json::detail::json_parser;
using cudf::json::detail::json_token;

template <int max_json_depth = 128>
std::vector<json_token> parse(std::string json_str,
                              bool single_quote,
                              bool control_char,
                              bool allow_tailing = true,
                              int max_string_len = 20000000,
                              int max_num_len    = 1000)
{
  cudf::get_json_object_options options;
  options.set_allow_single_quotes(single_quote);
  options.set_allow_unescaped_control_chars(control_char);
  options.set_allow_tailing_sub_string(allow_tailing);
  options.set_max_string_len(max_string_len);
  options.set_max_num_len(max_num_len);
  json_parser<max_json_depth> parser(options, json_str.data(), json_str.size());
  std::vector<json_token> tokens;
  json_token token = parser.next_token();
  tokens.push_back(token);
  while (token != json_token::ERROR && token != json_token::SUCCESS) {
    token = parser.next_token();
    tokens.push_back(token);
  }
  return tokens;
}

void test_basic(bool allow_single_quote, bool allow_control_char)
{
  std::vector<std::pair<std::string, std::vector<json_token>>> cases = {
    std::make_pair(
      // test terminal number
      std::string{"  \r\n\t  \r\n\t  1   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_NUMBER_INT, json_token::SUCCESS}),
    std::make_pair(
      // test terminal float
      std::string{"  \r\n\t  \r\n\t  1.5   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_NUMBER_FLOAT, json_token::SUCCESS}),
    std::make_pair(
      // test terminal string
      std::string{"  \r\n\t  \r\n\t  \"abc\"   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_STRING, json_token::SUCCESS}),
    std::make_pair(
      // test terminal true
      std::string{"  \r\n\t  \r\n\t  true   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_TRUE, json_token::SUCCESS}),
    std::make_pair(
      // test terminal false
      std::string{"  \r\n\t  \r\n\t  false   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_FALSE, json_token::SUCCESS}),
    std::make_pair(
      // test terminal null
      std::string{"  \r\n\t  \r\n\t  null   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_NULL, json_token::SUCCESS}),

    std::make_pair(
      // test numbers
      std::string{R"(    
            [
              0, 102, -0, -102, 0.3, -0.3000, 1e-050, -1e-5, 1.0e-5, -1.0010e-050, 1E+5, 1e0, 1E0, 1.3e5, -1e01, 1e00000
            ]
          )"},
      std::vector{json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::END_ARRAY,
                  json_token::SUCCESS}),
    std::make_pair(
      // test string
      std::string{"\"美国,中国\\u12f3\\u113E---abc---\\\", \\/, \\\\, \\b, \\f, \\n, \\r, \\t\""},
      std::vector{json_token::VALUE_STRING, json_token::SUCCESS}),
    std::make_pair(
      // test empty object
      std::string{"  {   }   "},
      std::vector{json_token::START_OBJECT, json_token::END_OBJECT, json_token::SUCCESS}),
    std::make_pair(
      // test empty array
      std::string{"   [   ]   "},
      std::vector{json_token::START_ARRAY, json_token::END_ARRAY, json_token::SUCCESS}),
    std::make_pair(
      // test nesting arrays
      std::string{R"(    
            [    
              1 ,    
              [    
                2 ,    
                [    
                  3 ,        
                  [    
                    41 , 42 , 43
                  ]
                ]
              ]
            ]
          )"},
      std::vector{json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::END_ARRAY,
                  json_token::END_ARRAY,
                  json_token::END_ARRAY,
                  json_token::END_ARRAY,
                  json_token::SUCCESS}),
    std::make_pair(
      // test nesting objects
      std::string{R"(    
            {    
              "k1" : "v1" ,    
              "k2" : {    
                "k3" : {    
                  "k4" : {    
                    "k51" : "v51" ,    
                    "k52" : "v52"    
                  }    
                }
              }
            }
          )"},
      std::vector{json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::FIELD_NAME,
                  json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::END_OBJECT,
                  json_token::END_OBJECT,
                  json_token::END_OBJECT,
                  json_token::END_OBJECT,
                  json_token::SUCCESS}),
    std::make_pair(
      // test nesting objects and arrays
      std::string{R"(    
            {
              "k1" : "v1",
              "k2" : [    
                1, {
                  "k21" : "v21",
                  "k22" : [1 , 2 , -1.5]
                }    
              ]
            }    
          )"},
      std::vector{json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::FIELD_NAME,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::FIELD_NAME,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::END_ARRAY,
                  json_token::END_OBJECT,
                  json_token::END_ARRAY,
                  json_token::END_OBJECT,
                  json_token::SUCCESS}),

    std::make_pair(
      // test invalid string: should have 4 HEX
      std::string{"\"  \\uFFF  \""},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid string: invalid HEX 'T'
      std::string{"\"  \\uTFFF  \""},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid string: unclosed string
      std::string{"  \"abc   "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid string: have no char after escape char '\'
      std::string{"\"\\"},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid string:  \X is not allowed
      std::string{"\" \\X   \""},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" +5 "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" 1.  "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" 1e  "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" 1e-  "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" infinity  "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{" {"},
      std::vector{json_token::START_OBJECT, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{" ["},
      std::vector{json_token::START_ARRAY, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{" {1} "},
      std::vector{json_token::START_OBJECT, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        {"k",}
      )"},
      std::vector{json_token::START_OBJECT, json_token::FIELD_NAME, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        {"k": }
      )"},
      std::vector{json_token::START_OBJECT, json_token::FIELD_NAME, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        {"k": 1 :}
      )"},
      std::vector{json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_NUMBER_INT,
                  json_token::ERROR}),

    std::make_pair(
      // test invalid structure
      std::string{R"(
        {"k": 1 , }
      )"},
      std::vector{json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_NUMBER_INT,
                  json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        [ 1 :
      )"},
      std::vector{json_token::START_ARRAY, json_token::VALUE_NUMBER_INT, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        [ 1,
      )"},
      std::vector{json_token::START_ARRAY, json_token::VALUE_NUMBER_INT, json_token::ERROR}),
    std::make_pair(
      // test invalid null
      std::string{" nul "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid false
      std::string{" fals "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid true
      std::string{" tru "},
      std::vector{json_token::ERROR}),

  };
  for (std::size_t i = 0; i < cases.size(); ++i) {
    std::string json_str                    = cases[i].first;
    std::vector<json_token> expected_tokens = cases[i].second;
    std::vector<json_token> actual_tokens = parse(json_str, allow_single_quote, allow_control_char);
    assert(actual_tokens == expected_tokens);
  }
}

void test_len_limitation()
{
  std::vector<std::string> v;
  v.push_back("  '123456'        ");
  v.push_back(
    "  'k\n\\'\\\"56'  ");  // do not count escape char '\', actual has 6 chars: k \n ' " 5 6
  v.push_back("  123456          ");
  v.push_back("  -1.23e-456      ");

  auto error_token = std::vector<json_token>{json_token::ERROR};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  true,  // control_char
                                                  true,  // allow_tailing
                                                  5,     // max_string_len
                                                  5);    // max_num_len
    assert(actual_tokens == error_token);
  }

  v.clear();
  v.push_back("   '12345'           ");
  v.push_back(
    "   'k\n\\'\\\"5'     ");  // do not count escape char '\', actual has 5 chars: k \n ' " 5
  auto expect_str_ret = std::vector<json_token>{json_token::VALUE_STRING, json_token::SUCCESS};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  true,  // control_char
                                                  true,  // allow_tailing
                                                  5,     // max_string_len
                                                  5);    // max_num_len
    assert(actual_tokens == expect_str_ret);
  }

  v.clear();
  v.push_back("    12345            ");
  v.push_back("    -1.23e-45        ");
  auto expect_num_ret = std::vector<json_token>{json_token::VALUE_NUMBER_INT, json_token::SUCCESS};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,   //  bool single_quote,
                                                  false,  // control_char
                                                  true,   // allow_tailing
                                                  5,      // max_string_len
                                                  5);     // max_num_len
    assert(actual_tokens[0] == json_token::VALUE_NUMBER_INT ||
           actual_tokens[0] == json_token::VALUE_NUMBER_FLOAT);
    assert(actual_tokens[1] == json_token::SUCCESS);
  }
}

void test_single_double_quote()
{
  std::vector<std::string> v;
  // allow \'  \" " in single quote
  v.push_back("'    \\\'     \\\"      \"          '");
  // allow \'  \"  ' in double quote
  v.push_back("\"   \\\' \\\"   '    \'      \"");  // C++ allow \' to represent ' in string
  auto expect_ret = std::vector<json_token>{json_token::VALUE_STRING, json_token::SUCCESS};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  false  // control_char
    );
    assert(actual_tokens == expect_ret);
  }

  v.clear();
  v.push_back("\"     \\'      \"");  // not allow \' when single_quote is disabled
  expect_ret = std::vector<json_token>{json_token::ERROR};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  false,  //  bool single_quote,
                                                  true    // control_char
    );
    assert(actual_tokens == expect_ret);
  }

  v.clear();
  v.push_back("\"     '   \\\"      \"");  // allow ' \" in double quote
  expect_ret = std::vector<json_token>{json_token::VALUE_STRING, json_token::SUCCESS};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  false,  //  bool single_quote,
                                                  true    // control_char
    );
    assert(actual_tokens == expect_ret);
  }

  v.clear();
  v.push_back("      'str'      ");  // ' is not allowed to quote string
  expect_ret = std::vector<json_token>{json_token::ERROR};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  false,  //  bool single_quote,
                                                  true    // control_char
    );
    assert(actual_tokens == expect_ret);
  }
}

void test_max_nested_len()
{
  std::vector<std::string> v;
  v.push_back("[[[[[]]]]]");
  v.push_back("{'k1':{'k2':{'k3':{'k4':{'k5': 5}}}}}");
  for (std::size_t i = 0; i < v.size(); ++i) {
    // set max nested len template value as 5
    std::vector<json_token> actual_tokens = parse<5>(v[i],
                                                     true,  //  bool single_quote,
                                                     true   // control_char
    );
    assert(actual_tokens[actual_tokens.size() - 1] == json_token::SUCCESS);
  }

  v.clear();
  v.push_back("[[[[[[]]]]]]");
  v.push_back("{'k1':{'k2':{'k3':{'k4':{'k5': {'k6': 6}}}}}}");
  for (std::size_t i = 0; i < v.size(); ++i) {
    // set max nested len template value as 5
    std::vector<json_token> actual_tokens = parse<5>(v[i],
                                                     true,  //  bool single_quote,
                                                     false  // control_char
    );
    assert(actual_tokens[actual_tokens.size() - 1] == json_token::ERROR);
  }
}

void test_control_char()
{
  std::vector<std::string> v;
  v.push_back("'   \t   \n   \b '");  // \t \n \b are control chars
  for (std::size_t i = 0; i < v.size(); ++i) {
    // set max nested len template value as 5
    std::vector<json_token> actual_tokens = parse<5>(v[i],
                                                     true,  //  bool single_quote,
                                                     true   // control_char
    );
    assert(actual_tokens[actual_tokens.size() - 1] == json_token::SUCCESS);
  }

  for (std::size_t i = 0; i < v.size(); ++i) {
    // set max nested len template value as 5
    std::vector<json_token> actual_tokens = parse<5>(v[i],
                                                     true,  //  bool single_quote,
                                                     false  // control_char
    );
    assert(actual_tokens[actual_tokens.size() - 1] == json_token::ERROR);
  }
}

void test_allow_tailing_useless_chars()
{
  std::vector<std::string> v;
  v.push_back("  0xxxx        ");  // 0 is valid JSON, tailing xxxx is ignored when allow tailing
  v.push_back("  {}xxxx  ");       // tailing xxxx is ignored
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  true,  // control_char
                                                  true   // allow_tailing is true
    );
    assert(actual_tokens[actual_tokens.size() - 1] == json_token::SUCCESS);
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  true,  // control_char
                                                  false  // allow_tailing is false
    );
    assert(actual_tokens[actual_tokens.size() - 1] == json_token::ERROR);
  }

  v.clear();
  v.push_back("    12345            ");
  v.push_back("    -1.23e-45        ");
  auto expect_num_ret = std::vector<json_token>{json_token::VALUE_NUMBER_INT, json_token::SUCCESS};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,   //  bool single_quote,
                                                  false,  // control_char
                                                  true,   // allow_tailing
                                                  5,      // max_string_len
                                                  5);     // max_num_len
    assert(actual_tokens[0] == json_token::VALUE_NUMBER_INT ||
           actual_tokens[0] == json_token::VALUE_NUMBER_FLOAT);
    assert(actual_tokens[1] == json_token::SUCCESS);
  }
}

void test_is_valid()
{
  std::string json_str = " {    \"k\"   :     [1,2,3]}   ";
  cudf::get_json_object_options options;
  json_parser<10> parser1(options, json_str.data(), json_str.size());
  assert(parser1.is_valid());

  json_str = " {[1,2,    ";
  json_parser<10> parser2(options, json_str.data(), json_str.size());
  assert(!parser2.is_valid());
}

TEST_F(JsonParserTests, NormalTest)
{
  test_basic(/*single_quote*/ true, /*control_char*/ true);
  test_basic(/*single_quote*/ true, /*control_char*/ false);
  test_basic(/*single_quote*/ false, /*control_char*/ true);
  test_basic(/*single_quote*/ false, /*control_char*/ false);
  test_len_limitation();
  test_single_double_quote();
  test_max_nested_len();
  test_control_char();
  test_allow_tailing_useless_chars();
  test_is_valid();
}
