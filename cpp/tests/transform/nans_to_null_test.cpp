/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <arrow/util/bit_util.h>
#include <arrow/testing/gtest_util.h>
#include <tests/transform/arrow_utils.hpp>

template <typename T>
struct NaNsToNullTest : public cudf::test::BaseFixture {
  void run_test(cudf::column_view const& input, cudf::column_view const& expected)
  {
    auto got_mask = cudf::nans_to_nulls(input);
    cudf::column got(input);
    got.set_null_mask(std::move(*(got_mask.first)));

    EXPECT_EQ(expected.null_count(), got_mask.second);

    cudf::test::expect_columns_equal(expected, got.view());
  }

  std::unique_ptr<cudf::column> create_expected(std::vector<T> const& input,
                                                std::vector<bool> const& mask = {})
  {
    std::vector<T> expected(input);
    std::vector<bool> expected_mask;

    if (mask.size() > 0) {
      std::transform(
        input.begin(),
        input.end(),
        mask.begin(),
        std::back_inserter(expected_mask),
        [](T val, bool validity) { return (std::isnan(val) or validity == false) ? false : true; });
    } else {
      std::transform(input.begin(), input.end(), std::back_inserter(expected_mask), [](T val) {
        return (std::isnan(val)) ? false : true;
      });
    }

    return cudf::test::fixed_width_column_wrapper<T>(
             expected.begin(), expected.end(), expected_mask.begin())
      .release();
  }
};

using test_types = ::testing::Types<float, double>;

TYPED_TEST_CASE(NaNsToNullTest, test_types);

TYPED_TEST(NaNsToNullTest, WithMask)
{
  using T = TypeParam;

  std::vector<T> input   = {1, NAN, 3, NAN, 5, NAN};
  std::vector<bool> mask = {1, 1, 1, 1, 0, 0};
  auto input_column =
    cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end(), mask.begin());
  auto expected_column = this->create_expected(input, mask);
  this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, WithNoMask)
{
  using T = TypeParam;

  std::vector<T> input = {1, NAN, 3, NAN, 5, NAN};
  auto input_column    = cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end());
  auto expected_column = this->create_expected(input);
  this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, NoNANWithMask)
{
  using T = TypeParam;

  std::vector<T> input   = {1, 2, 3, 4, 5, 6};
  std::vector<bool> mask = {1, 1, 1, 1, 0, 0};
  auto input_column =
    cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end(), mask.begin());
  auto expected_column = this->create_expected(input, mask);
  this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, NoNANNoMask)
{
  using T = TypeParam;

  std::vector<T> input = {1, 2, 3, 4, 5, 6};
  auto input_column    = cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end());
  auto expected_column = this->create_expected(input);
  this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, EmptyColumn)
{
  using T = TypeParam;

  std::vector<T> input = {};
  auto input_column    = cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end());
  auto expected_column = this->create_expected(input);
  this->run_test(input_column, expected_column->view());
}

struct NaNsToNullFailTest : public cudf::test::BaseFixture {
};

TEST_F(NaNsToNullFailTest, StringType)
{
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
  cudf::test::strings_column_wrapper input(strings.begin(), strings.end());

  EXPECT_THROW(cudf::nans_to_nulls(input), cudf::logic_error);
}

TEST_F(NaNsToNullFailTest, IntegerType)
{
  std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
  auto input_column = cudf::test::fixed_width_column_wrapper<int32_t>(input.begin(), input.end());

  EXPECT_THROW(cudf::nans_to_nulls(input_column), cudf::logic_error);
}

TEST_F(NaNsToNullFailTest, ToArrow){
  std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
  auto col1 = cudf::test::fixed_width_column_wrapper<int32_t>(input.begin(), input.end());
  std::vector<int32_t> input2 = {1, 2, 7, 4, 5, 6};
  std::vector<bool> mask = {1, 1, 1, 1, 0, 0};
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>(input2.begin(), input2.end(), mask.begin());
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  cudf::test::lists_column_wrapper<int> col3{{{0, 1, 2, 3}, valids}, {{4, 6}, valids}, {{4, 6}, valids}, {{7, 11}, valids}, {{9, 18}, valids}, {{1, 2}, valids}};
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of"};
  cudf::test::strings_column_wrapper col4(strings.begin(), strings.end());
   cudf::test::strings_column_wrapper str(
    {"fff", "aaa", "", "fff", "ccc", "aaa"});
  auto col5 = cudf::dictionary::encode(str);

  cudf::table_view input_table({col1, col2, col3, col4, col5->view()}); 

  auto arrow_table = cudf::to_arrow(input_table, {"a", "b", "c", "d", "c"});
  arrow::PrettyPrint(*arrow_table, arrow::PrettyPrintOptions{}, &std::cout);
}

TEST_F(NaNsToNullFailTest, Fromarrow){

    std::vector<int64_t> data1 = {1, 2, 3, 4, 5, 6, 8};
    std::vector<int32_t> data2 = {1, 2, 3, 4, 5, 6, 8};
    auto int64array = std::make_shared<arrow::Int64Array>(7, arrow::Buffer::Wrap(data1));
    auto int32array = std::make_shared<arrow::Int32Array>(7, arrow::Buffer::Wrap(data2), arrow::BitUtil::BytesToBits(std::vector<uint8_t>{0, 1, 0, 1, 0, 1, 1}).ValueOrDie());
    std::shared_ptr<arrow::StringArray> str_ar;
    arrow::StringBuilder str_builder;
    str_builder.AppendValues({"Hi", "I", "am", "from", "cudf", "team", "."});
    str_builder.Finish(&str_ar);

    std::vector<int64_t> ind = {0, 1, 2, 1, 2, 0, 2};
    auto indices = std::make_shared<arrow::Int32Array>(7, arrow::Buffer::Wrap(ind), arrow::BitUtil::BytesToBits(std::vector<uint8_t>{0, 1, 0, 1, 0, 1, 1}).ValueOrDie());
    std::shared_ptr<arrow::StringArray> key_ar;
    str_builder.Reset();
    str_builder.AppendValues({"Hi", "I", "am"});
    str_builder.Finish(&key_ar);

    auto dict_ar = std::make_shared<arrow::DictionaryArray>(arrow::dictionary(arrow::int32(), arrow::utf8()), indices, key_ar);
    
    std::vector<int32_t> offset = {0, 2, 5, 6, 7, 8, 10, 11};
    std::vector<int32_t> list_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto data = std::make_shared<arrow::Int32Array>(list_data.size(), arrow::Buffer::Wrap(list_data));

    std::cout<<"Before creating list"<<std::endl;
    auto list_ar = std::make_shared<arrow::ListArray>(arrow::list(data->type()), static_cast<int64_t>(7), arrow::Buffer::Wrap(offset), data);
    arrow::PrettyPrint(*list_ar, arrow::PrettyPrintOptions{}, &std::cout);
    std::cout<<"After creating list"<<std::endl;

    std::vector <std::shared_ptr<arrow::Field>> schema_vector = {arrow::field("a", int32array->type()), arrow::field("b", int64array->type()), arrow::field("c", str_ar->type()), arrow::field("d", dict_ar->type()), arrow::field("e", list_ar->type())};
    auto schema = std::make_shared<arrow::Schema>(schema_vector);
    auto table = arrow::Table::Make(schema, {int32array, int64array, str_ar, dict_ar, list_ar});
    arrow::PrettyPrint(*table, arrow::PrettyPrintOptions{}, &std::cout);
    std::unique_ptr<cudf::table> cudf_table = cudf::arrow_to_table(table);
    auto t_view = cudf_table->view();

    for (auto const& c : t_view) {
        cudf::test::print(c);
    }
}

TEST_F(NaNsToNullFailTest, FromArrow){
    auto int64array = get_arrow_array<int64_t>({1, 2, 3, 4, 5}, {1, 0, 1, 0, 1});
    auto int32array = get_arrow_array<int32_t>({1, 2, 3, 4, 5});
    auto string_array = get_arrow_array<cudf::string_view>({"Hi", "I", "am", "Calling", "new Api"});
    auto dict_array = get_arrow_dict_array({1, 2, 3}, {0, 1, 2, 1, 0}, {1, 0, 1, 0, 1});
    auto list_array = get_arrow_list_array({0, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 5, 6, 7, 8});

    std::vector <std::shared_ptr<arrow::Field>> schema_vector = {arrow::field("a", int32array->type()), arrow::field("b", int64array->type()), arrow::field("c", string_array->type()), arrow::field("d", dict_array->type()), arrow::field("e", list_array->type())};

    auto schema = std::make_shared<arrow::Schema>(schema_vector);
    auto table = arrow::Table::Make(schema, {int32array, int64array, string_array, dict_array, list_array});
    arrow::PrettyPrint(*table, arrow::PrettyPrintOptions{}, &std::cout);
}
