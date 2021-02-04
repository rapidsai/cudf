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

#include <cudf/copying.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

namespace cudf {
namespace test {

struct PackUnpackTest : public BaseFixture {
  void run_test(cudf::table_view const& t)
  {
    // verify pack/unpack works
    auto packed   = pack(t);
    auto unpacked = unpack(packed);
    cudf::test::expect_tables_equal(t, unpacked);

    // verify pack_metadata itself works
    auto metadata = pack_metadata(
      unpacked, reinterpret_cast<uint8_t const*>(packed.gpu_data->data()), packed.gpu_data->size());
    EXPECT_EQ(metadata.size(), packed.metadata_->size());
    EXPECT_EQ(
      std::equal(metadata.data(), metadata.data() + metadata.size(), packed.metadata_->data()),
      true);
  }
  void run_test(std::vector<column_view> const& t) { run_test(cudf::table_view{t}); }
};

// clang-format off
TEST_F(PackUnpackTest, SingleColumnFixedWidth)
{
  fixed_width_column_wrapper<int64_t> col1 ({ 1, 2, 3, 4, 5, 6, 7},
                                            { 1, 1, 1, 0, 1, 0, 1});

  this->run_test({col1});
}

TEST_F(PackUnpackTest, SingleColumnFixedWidthNonNullable)
{
  fixed_width_column_wrapper<int64_t> col1 ({ 1, 2, 3, 4, 5, 6, 7});

  this->run_test({col1});
}

TEST_F(PackUnpackTest, MultiColumnFixedWidth)
{
  fixed_width_column_wrapper<int16_t> col1 ({ 1, 2, 3, 4, 5, 6, 7},
                                            { 1, 1, 1, 0, 1, 0, 1});
  fixed_width_column_wrapper<float>   col2 ({ 7, 8, 6, 5, 4, 3, 2},
                                            { 1, 0, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper<double>  col3 ({ 8, 4, 2, 0, 7, 1, 3},
                                            { 0, 1, 1, 1, 1, 1, 1});

  this->run_test({col1, col2, col3});
}

TEST_F(PackUnpackTest, MultiColumnWithStrings)
{
  fixed_width_column_wrapper<int16_t> col1 ({ 1, 2, 3, 4, 5, 6, 7},
                                            { 1, 1, 1, 0, 1, 0, 1});
  strings_column_wrapper              col2 ({"Lorem", "ipsum", "dolor", "sit", "amet", "ort", "ral"},
                                            {      1,       0,       1,     1,      1, 0,     1});
  strings_column_wrapper              col3 ({"", "this", "is", "a", "column", "of", "strings"});

  this->run_test({col1, col2, col3});
}

TEST_F(PackUnpackTest, EmptyColumns)
{
  {
    auto empty_string = cudf::strings::detail::make_empty_strings_column();    
    cudf::table_view src_table({static_cast<cudf::column_view>(*empty_string)});
    this->run_test(src_table);
  }
  
  {
    cudf::test::strings_column_wrapper str{"abc"};
    auto empty_string = cudf::empty_like(str);    
    cudf::table_view src_table({static_cast<cudf::column_view>(*empty_string)});
    this->run_test(src_table);
  }

  {
    cudf::test::fixed_width_column_wrapper<int> col0;
    cudf::test::dictionary_column_wrapper<int> col1;
    cudf::test::strings_column_wrapper col2;
    cudf::test::lists_column_wrapper<int> col3;
    cudf::test::structs_column_wrapper col4({});

    cudf::table_view src_table({col0, col1, col2, col3, col4});
    this->run_test(src_table);
  }
}

std::vector<std::unique_ptr<column>> generate_lists(bool include_validity)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  if(include_validity){
    auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
    cudf::test::lists_column_wrapper<int> list0{{1, 2, 3},
                                          {4, 5},
                                          {6},
                                          {{7, 8}, valids},
                                          {9, 10, 11},
                                          LCW{},
                                          LCW{},
                                          {{-1, -2, -3, -4, -5}, valids},
                                          {{100, -200}, valids}};

    cudf::test::lists_column_wrapper<int> list1{{{{1, 2, 3}, valids}, {4, 5}},
                                          {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                          {{LCW{6}}},                                                                                    
                                          {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                          {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                          {{LCW{}}},
                                          {{-10}, {-100, -200}},
                                          {{-10, -200}, LCW{}, {8, 9}},
                                          {LCW{8}, LCW{}, LCW{9}, {5, 6}}};

    std::vector<std::unique_ptr<column>> out;
    out.push_back(list0.release());
    out.push_back(list1.release());
    return out;
  }
  
  cudf::test::lists_column_wrapper<int> list0{{1, 2, 3},
                                        {4, 5},
                                        {6},
                                        {7, 8},
                                        {9, 10, 11},
                                        LCW{},
                                        LCW{},
                                        {-1, -2, -3, -4, -5},
                                        {-100, -200}};

  cudf::test::lists_column_wrapper<int> list1{{{1, 2, 3}, {4, 5}},
                                        {LCW{}, LCW{}, {7, 8}, LCW{}},
                                        {{LCW{6}}},                                                                                    
                                        {{7, 8}, {9, 10, 11}, LCW{}},
                                        {LCW{}, {-1, -2, -3, -4, -5}},
                                        {{LCW{}}},
                                        {{-10}, {-100, -200}},
                                        {{-10, -200}, LCW{}, {8, 9}},
                                        {LCW{8}, LCW{}, LCW{9}, {5, 6}}};

  std::vector<std::unique_ptr<column>> out;
  out.push_back(list0.release());
  out.push_back(list1.release());
  return out;
}

std::vector<std::unique_ptr<column>> generate_structs(bool include_validity)
{
  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  std::vector<bool> names_validity{1, 1, 1, 1, 1, 1, 1, 1, 1};
  strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {1, 1, 1, 1, 0, 1, 0, 0, 1};
  auto ages_column                = include_validity ? fixed_width_column_wrapper<int>(
                                          ages.begin(), ages.end(), ages_validity.begin())
                                      : fixed_width_column_wrapper<int>(ages.begin(), ages.end());

  // 3. Boolean "is_human" column.
  std::vector<bool> is_human{true, true, false, false, false, false, true, true, true};
  std::vector<bool> is_human_validity{1, 1, 1, 0, 1, 1, 1, 1, 0};
  auto is_human_col = include_validity
                        ? fixed_width_column_wrapper<bool>(
                            is_human.begin(), is_human.end(), is_human_validity.begin())
                        : fixed_width_column_wrapper<bool>(is_human.begin(), is_human.end());

  // Assemble struct column.
  auto const struct_validity = std::vector<bool>{1, 1, 1, 1, 1, 0, 0, 1, 0};
  auto struct_column =
    include_validity
      ? structs_column_wrapper({names_column, ages_column, is_human_col}, struct_validity.begin())
      : structs_column_wrapper({names_column, ages_column, is_human_col});

  std::vector<std::unique_ptr<column>> out;
  out.push_back(struct_column.release());
  return out;
}

std::vector<std::unique_ptr<column>> generate_struct_of_list()
{
  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  std::vector<bool> names_validity{1, 1, 1, 1, 1, 1, 1, 1, 1};
  strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {1, 1, 1, 1, 0, 1, 0, 0, 1};
  auto ages_column =
    fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // 3. List column
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;
  std::vector<bool> list_validity{1, 1, 1, 1, 1, 0, 1, 0, 1};
  lists_column_wrapper<cudf::string_view> list({{{"abc", "d", "edf"}, {"jjj"}},
                                                {{"dgaer", "-7"}, LCW{}},
                                                {{LCW{}}},
                                                {{"qwerty"}, {"ral", "ort", "tal"}, {"five", "six"}},
                                                {LCW{}, LCW{}, {"eight", "nine"}},
                                                {{LCW{}}},
                                                {{"fun"}, {"a", "bc", "def", "ghij", "klmno", "pqrstu"}},
                                                {{"seven", "zz"}, LCW{}, {"xyzzy"}},
                                                {{LCW{"negative 3", "  ", "cleveland"}}} },
                                            list_validity.begin());

  // Assemble struct column.
  auto const struct_validity = std::vector<bool>{1, 1, 1, 1, 1, 0, 0, 1, 0};
  auto struct_column =
    structs_column_wrapper({names_column, ages_column, list}, struct_validity.begin());

  std::vector<std::unique_ptr<column>> out;
  out.push_back(struct_column.release());
  return out;
}

std::vector<std::unique_ptr<column>> generate_list_of_struct()
{
  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin",
    "Abc", "Def", "Xyz", "Five", "Seventeen", "Dol", "Est"};
  std::vector<bool> names_validity{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1};
  strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102, -1, -2, -3, -4, -5, -6, -7};
  std::vector<bool> ages_validity = {1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1};
  auto ages_column =
    fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // Assemble struct column.
  auto const struct_validity = std::vector<bool>{1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1};
  auto struct_column =
    structs_column_wrapper({names_column, ages_column}, struct_validity.begin());


  // 3. List column
  std::vector<bool> list_validity{1, 1, 1, 1, 1, 0, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<int> offsets{0, 1, 4, 5, 7, 7, 10, 13, 14, 16};
  auto list = cudf::make_lists_column(9, offsets.release(), struct_column.release(), 
                                      2, cudf::test::detail::make_null_mask(list_validity.begin(), list_validity.begin() + 9));

  std::vector<std::unique_ptr<column>> out;
  out.push_back(std::move(list));
  return out;
}

TEST_F(PackUnpackTest, Lists)
{  
  // lists
  {      
    auto cols = generate_lists(false);
    std::vector<column_view> col_views;
    std::transform(cols.begin(), cols.end(), std::back_inserter(col_views), [](std::unique_ptr<column> const& col){
      return static_cast<column_view>(*col);
    });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }

  // lists with validity
  {
    auto cols = generate_lists(true);
    std::vector<column_view> col_views;
    std::transform(cols.begin(), cols.end(), std::back_inserter(col_views), [](std::unique_ptr<column> const& col){
      return static_cast<column_view>(*col);
    });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }    
}

TEST_F(PackUnpackTest, Structs)
{
  // structs
  {      
    auto cols = generate_structs(false);
    std::vector<column_view> col_views;
    std::transform(cols.begin(), cols.end(), std::back_inserter(col_views), [](std::unique_ptr<column> const& col){
      return static_cast<column_view>(*col);
    });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }

  // structs with validity
  {
    auto cols = generate_structs(true);
    std::vector<column_view> col_views;
    std::transform(cols.begin(), cols.end(), std::back_inserter(col_views), [](std::unique_ptr<column> const& col){
      return static_cast<column_view>(*col);
    });
    cudf::table_view src_table(col_views);
    this->run_test(src_table);
  }
}

TEST_F(PackUnpackTest, NestedTypes)
{ 
  // build one big table containing, lists, structs, structs<list>, list<struct>
  std::vector<column_view> col_views;
  
  auto lists = generate_lists(true);
  std::transform(lists.begin(), lists.end(), std::back_inserter(col_views), [](std::unique_ptr<column> const& col){
    return static_cast<column_view>(*col);
  });
  
  auto structs = generate_structs(true);    
  std::transform(structs.begin(), structs.end(), std::back_inserter(col_views), [](std::unique_ptr<column> const& col){
    return static_cast<column_view>(*col);
  });

  auto struct_of_list = generate_struct_of_list();
  std::transform(struct_of_list.begin(), struct_of_list.end(), std::back_inserter(col_views), [](std::unique_ptr<column> const& col){
    return static_cast<column_view>(*col);
  });

  auto list_of_struct = generate_list_of_struct();
  std::transform(list_of_struct.begin(), list_of_struct.end(), std::back_inserter(col_views), [](std::unique_ptr<column> const& col){
    return static_cast<column_view>(*col);
  });

  cudf::table_view src_table(col_views);
  this->run_test(src_table);
}

TEST_F(PackUnpackTest, NestedEmpty)
{
  // this produces an empty strings column with no children,
  // nested inside a list
  {
    auto empty_string = cudf::strings::detail::make_empty_strings_column();
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_string), 0, rmm::device_buffer{0});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty strings column with children that have no data,
  // nested inside a list
  {
    cudf::test::strings_column_wrapper str{"abc"};
    auto empty_string = cudf::empty_like(str);
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_string), 0, rmm::device_buffer{0});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list       = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_list), 0, rmm::device_buffer{0});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list       = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_list), 0, rmm::device_buffer{0});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }

  // this produces an empty struct column with children that have no data,
  // nested inside a list
  {
    cudf::test::fixed_width_column_wrapper<int> ints{0, 1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<float> floats{4, 3, 2, 1, 0};
    auto struct_column = cudf::test::structs_column_wrapper({ints, floats});
    auto empty_struct  = cudf::empty_like(struct_column);
    auto offsets       = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list          = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_struct), 0, rmm::device_buffer{0});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});
    this->run_test(src_table);
  }
}
// clang-format on

}  // namespace test
}  // namespace cudf