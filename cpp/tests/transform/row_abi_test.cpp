/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/errc.hpp>
#include <cudf/join/join.hpp>
#include <cudf/row_operators.cuh>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream.hpp>

#include <ast/jit/expressions.hpp>
#include <cudf_test_fragments.hpp>
#include <jit/cache.hpp>
#include <row_operator/type_names.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

using cudf::test::fixed_width_column_wrapper;
struct RowABI : cudf::test::BaseFixture {};
static_assert(sizeof(cudf::abi_types::expected<cudf::abi_types::i64, cudf::abi_types::i32>) == 16);
using expected_i64 = cudf::abi_types::expected<cudf::abi_types::i64, cudf::abi_types::i32>;
static_assert(offsetof(expected_i64, error) == 8);
static_assert(sizeof(cudf::abi_types::i32) == 4);
static_assert(sizeof(cudf::abi_types::i64) == 8);
static_assert(sizeof(cudf::abi_types::decimal128) == 32);
static_assert(alignof(cudf::abi_types::decimal128) == 16);
static_assert(sizeof(cudf::abi_types::string_view) == 16);
static_assert(offsetof(cudf::abi_types::optional<cudf::abi_types::i64>, validity) == 8);

static_assert(std::is_same_v<decltype(cudf::row_abi::cudf_row_op_ADD_i32_i32),
                             cudf::abi_types::i32(cudf::abi_types::i32, cudf::abi_types::i32)>);
static_assert(std::is_same_v<decltype(cudf::row_abi::cudf_row_op_ADD_OVERFLOW_i32_i32),
                             cudf::abi_types::expected<cudf::abi_types::i32, cudf::row_abi::status>(
                               cudf::abi_types::i32, cudf::abi_types::i32)>);
static_assert(
  std::is_same_v<decltype(cudf::row_abi::cudf_row_op_IS_NULL_i32),
                 cudf::abi_types::bool8(cudf::abi_types::optional<cudf::abi_types::i32>)>);
static_assert(std::is_same_v<decltype(cudf::row_abi::cudf_row_op_COALESCE_i32_i32),
                             cudf::abi_types::optional<cudf::abi_types::i32>(
                               cudf::abi_types::optional<cudf::abi_types::i32>,
                               cudf::abi_types::optional<cudf::abi_types::i32>)>);

namespace {

std::string const optional_stores = R"(
#define STORE_OPTIONAL(T) \
__device__ status store_optional_##T(void const* columns,i32 col,i32 row,optional<T> value) { \
  set_validity(columns,col,row,value.validity); \
  return value.validity ? set_##T(columns,col,row,value.value) : success; }
STORE_OPTIONAL(i32)
STORE_OPTIONAL(decimal32)
#undef STORE_OPTIONAL
__device__ status store_optional_list_i32(void const* column,i32 row,i32 element,optional<i32> value) {
  set_validity(column,0,row,element,value.validity);
  return value.validity ? set_list_i32(column,row,element,value.value) : success;
}
template<typename O>
__device__ status store_optional_string_view(void const* columns,i32 col,i32 row,optional<string_view> value,type_tag<O> tag) {
  set_validity(columns,col,row,value.validity);
  return value.validity ? set_string_view(columns,col,row,value.value,tag) : success;
}
template<typename O>
__device__ status store_optional_list_string_view(void const* column,i32 row,i32 element,optional<string_view> value,type_tag<O> tag) {
  set_validity(column,0,row,element,value.validity);
  return value.validity ? set_list_string_view(column,row,element,value.value,tag) : success;
}
template<typename I>
__device__ status store_optional_dict_i32(void const* columns,i32 col,i32 row,optional<dictionary_index> value,index_type_tag<I> tag) {
  set_validity(columns,col,row,value.validity);
  return value.validity ? set_dict_i32(columns,col,row,value.value,tag) : success;
}
template<typename I>
__device__ status store_optional_dict_list_i32(void const* columns,i32 col,i32 row,optional<dictionary_index> value,index_type_tag<I> tag) {
  set_validity(columns,col,row,value.validity);
  return value.validity ? set_dict_list_i32(columns,col,row,value.value,tag) : success;
}
)";

// Preserve the schema's physical index width. Public dictionary factories
// normalize unsigned indices to signed types, so construct those codec fixtures
// directly rather than silently exercising a different overload.
std::unique_ptr<cudf::column> make_codec_dictionary(std::unique_ptr<cudf::column> keys,
                                                    std::unique_ptr<cudf::column> indices,
                                                    cuda::device_buffer<std::byte>&& mask,
                                                    cudf::size_type null_count)
{
  auto const rows = indices->size();
  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::move(indices));
  children.emplace_back(std::move(keys));
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::DICTIONARY32},
                                        rows,
                                        rmm::device_buffer{},
                                        std::move(mask),
                                        null_count,
                                        std::move(children));
}

void apply_source(std::string const& source,
                  std::span<cudf::transform_input const> inputs,
                  cudf::mutable_table_view outputs,
                  std::optional<void*> data = std::nullopt,
                  cuda::stream_ref stream   = cudf::get_default_stream())
{
  cudf::transform(
    source, cudf::udf_source_type::CUDA, cudf::null_aware::YES, data, inputs, outputs, stream);
}
}  // namespace
TEST_F(RowABI, SlicesScalarsAndIndependentValidity)
{
  std::string const udf = optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in, void const* out, void*, i32 row) {
      auto input=load_i32_nullable(in,0,row);
      auto scalar=load_i32_nullable(in,1,0);
      auto first=input.validity ? optional<i32>{input.value+scalar.value,1} : optional<i32>{};
      auto error=store_optional_i32(out,0,row,first);
      if(error!=success) return error;
      return store_optional_i32(out,1,row,input.validity ? optional<i32>{} : optional<i32>{row,1});
    })";
  for (int rows : {0, 1, 31, 32, 33, 65}) {
    std::vector<int32_t> values(rows + 4, 5), zeros(rows + 4, -9);
    std::vector<bool> validity(rows + 4, true), first_valid(rows + 4, true),
      second_valid(rows + 4, true);
    std::vector<int32_t> first_expected = zeros, second_expected = zeros;
    for (int row = 0; row < rows; ++row) {
      bool valid               = row % 5 != 0;
      validity[row + 2]        = valid;
      first_valid[row + 2]     = valid;
      second_valid[row + 2]    = !valid;
      first_expected[row + 2]  = 12;
      second_expected[row + 2] = row;
    }
    fixed_width_column_wrapper<int32_t> input(values.begin(), values.end(), validity.begin());
    fixed_width_column_wrapper<int32_t> scalar{7};
    auto first = fixed_width_column_wrapper<int32_t>(
                   zeros.begin(), zeros.end(), cudf::test::iterators::no_nulls())
                   .release();
    auto second        = std::make_unique<cudf::column>(*first);
    auto sliced_input  = cudf::slice(input, {2, rows + 2}).front();
    auto sliced_first  = cudf::mutable_column_view{first->type(),
                                                  rows,
                                                  first->mutable_view().data<int32_t>(),
                                                  first->mutable_view().null_mask(),
                                                  0,
                                                  2};
    auto sliced_second = cudf::mutable_column_view{second->type(),
                                                   rows,
                                                   second->mutable_view().data<int32_t>(),
                                                   second->mutable_view().null_mask(),
                                                   0,
                                                   2};
    std::array<cudf::transform_input, 2> inputs{sliced_input, cudf::scalar_column_view{scalar}};
    rmm::cuda_stream stream;
    apply_source(udf,
                 inputs,
                 cudf::mutable_table_view{{sliced_first, sliced_second}},
                 std::nullopt,
                 static_cast<cuda::stream_ref>(stream));
    first->set_null_count(std::ranges::count(first_valid, false));
    second->set_null_count(std::ranges::count(second_valid, false));
    fixed_width_column_wrapper<int32_t> expected_first(
      first_expected.begin(), first_expected.end(), first_valid.begin());
    fixed_width_column_wrapper<int32_t> expected_second(
      second_expected.begin(), second_expected.end(), second_valid.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_first, first->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_second, second->view());
  }
}

TEST_F(RowABI, NullableListParentBypassesUnusedChild)
{
  std::array<bool, 3> validity{true, false, true};
  auto input =
    cudf::test::lists_column_wrapper<int32_t>({{1}, {}, {3}}, validity.begin()).release();
  auto output =
    fixed_width_column_wrapper<int32_t>({0, 0, 0}, cudf::test::iterators::no_nulls()).release();
  auto sizes       = std::make_unique<cudf::column>(*output);
  auto list_output = std::make_unique<cudf::column>(*input);
  std::array<cudf::transform_input, 1> inputs{input->view()};
  apply_source(optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto column=cudf_row_input_column(in,0);
      auto element = row==1 ? 2147483647 : 0;
      auto value=load_list_i32_nullable(column,row,element);
      auto size=load_list_size_nullable(column,row);
      auto error=store_optional_i32(out,0,row,value);
      if(error!=success) return error;
      error=store_optional_i32(out,1,row,size);
      if(error!=success) return error;
      return store_optional_list_i32(cudf_row_output_column(out,2),row,element,value);
    })",
               inputs,
               cudf::mutable_table_view{
                 {output->mutable_view(), sizes->mutable_view(), list_output->mutable_view()}});
  output->set_null_count(1);
  sizes->set_null_count(1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({1, 0, 3}, {1, 0, 1}),
                                 output->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({1, 0, 1}, {1, 0, 1}),
                                 sizes->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input->view(), list_output->view());
}

TEST_F(RowABI, NullableDictionaryAndListCodecs)
{
  auto check = []<typename Index>() {
    auto keys = fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 0, 1}}.release();
    std::array<bool, 3> key_validity{true, false, true};
    auto list_keys =
      cudf::test::lists_column_wrapper<int32_t>({{10}, {20}, {30}}, key_validity.begin()).release();
    auto indices =
      fixed_width_column_wrapper<Index>{
        {Index{2}, Index{1}, Index{0}, std::numeric_limits<Index>::max()}}
        .release();
    auto mask       = fixed_width_column_wrapper<int32_t>{{0, 0, 0, 0}, {1, 1, 1, 0}}.release();
    auto dictionary = make_codec_dictionary(std::move(keys),
                                            std::make_unique<cudf::column>(*indices),
                                            cudf::copy_bitmask(mask->view()),
                                            1);
    auto dictionary_list = make_codec_dictionary(
      std::move(list_keys), std::move(indices), cudf::copy_bitmask(mask->view()), 1);
    // Nullable keys require the nullable codec even without a parent mask.
    auto no_parent_mask =
      cudf::column_view{dictionary->type(),
                        3,
                        nullptr,
                        nullptr,
                        0,
                        0,
                        {dictionary->child(0).view(), dictionary->child(1).view()}};
    auto reference  = cudf::ast::column_reference{0};
    auto ast_output = cudf::compute_column_jit(cudf::table_view{{no_parent_mask}}, reference);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({30, 0, 10}, {1, 0, 1}),
                                   ast_output->view());
    auto dictionary_output = std::make_unique<cudf::column>(*dictionary);
    auto list_output       = std::make_unique<cudf::column>(*dictionary_list);
    auto output =
      fixed_width_column_wrapper<int32_t>({0, 0, 0, 0}, cudf::test::iterators::no_nulls())
        .release();
    auto second = std::make_unique<cudf::column>(*output);
    auto sizes  = std::make_unique<cudf::column>(*output);
    std::array<cudf::transform_input, 2> inputs{dictionary->view(), dictionary_list->view()};
    auto const tag = cudf::jit::row_abi_type(cudf::data_type{cudf::type_to_id<Index>()});
    auto source    = "#define INDEX " + tag + optional_stores + R"(
      extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
        auto first=load_dict_i32_nullable(in,0,row,index_type_tag<INDEX>{});
        auto column=cudf_row_input_column(in,1);
        auto count=load_dict_list_size_nullable(column,row,index_type_tag<INDEX>{});
        auto second=load_dict_list_i32_nullable(column,row,0,index_type_tag<INDEX>{});
        auto error=store_optional_i32(out,0,row,first);
        if(error!=success) return error;
        error=store_optional_i32(out,1,row,second);
        if(error!=success) return error;
        error=store_optional_i32(out,2,row,count);
        if(error!=success) return error;
        optional<dictionary_index> index = row<3 ? optional<dictionary_index>{2-row,1} : optional<dictionary_index>{};
        error=store_optional_dict_i32(out,3,row,index,index_type_tag<INDEX>{});
        if(error!=success) return error;
        return store_optional_dict_list_i32(out,4,row,index,index_type_tag<INDEX>{});
      })";
    apply_source(source,
                 inputs,
                 cudf::mutable_table_view{{output->mutable_view(),
                                           second->mutable_view(),
                                           sizes->mutable_view(),
                                           dictionary_output->mutable_view(),
                                           list_output->mutable_view()}});
    output->set_null_count(2);
    second->set_null_count(2);
    sizes->set_null_count(2);
    fixed_width_column_wrapper<int32_t> expected{{30, 0, 10, 0}, {1, 0, 1, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, second->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({1, 0, 1, 0}, {1, 0, 1, 0}),
                                   sizes->view());
    // The dictionary comparator assumes nonnullable scalar keys; compare codec storage.
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(dictionary->child(0).view(), dictionary_output->child(0).view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(dictionary->child(1).view(), dictionary_output->child(1).view());
    auto const scalar_parent_mask = cudf::column_view{mask->type(),
                                                      mask->size(),
                                                      mask->view().head<int32_t>(),
                                                      dictionary_output->view().null_mask(),
                                                      1};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(mask->view(), scalar_parent_mask);
    // Dictionary-list equality is not supported by the column comparison helper.
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(dictionary_list->child(0).view(), list_output->child(0).view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(dictionary_list->child(1).view(), list_output->child(1).view());
    auto const parent_mask = cudf::column_view{
      mask->type(), mask->size(), mask->view().head<int32_t>(), list_output->view().null_mask(), 1};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(mask->view(), parent_mask);
  };
  check.operator()<int8_t>();
  check.operator()<int16_t>();
  check.operator()<int32_t>();
  check.operator()<int64_t>();
  check.operator()<uint8_t>();
  check.operator()<uint16_t>();
  check.operator()<uint32_t>();
  check.operator()<uint64_t>();
}

TEST_F(RowABI, ExternalFragmentAndRecovery)
{
  auto range = cudf_test_fragments::file_ranges[cudf_test_fragments::row_abi_increment];
  auto udf   = cudf_test_fragments::files.subspan(range[0], range[1]);
  fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  auto output = fixed_width_column_wrapper<int32_t>{0, 0, 0}.release();
  std::array<cudf::transform_input, 1> inputs{input};
  auto run = [&] {
    cudf::transform_lto(udf,
                        cudf::lto_binary_type::FATBIN,
                        cudf::null_aware::YES,
                        std::nullopt,
                        inputs,
                        cudf::mutable_table_view{{output->mutable_view()}});
  };
  run();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({2, 3, 4}), output->view());
  std::array<cudf::transform_output, 1> specs{
    {{cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::ALL_VALID}}};
  auto allocated = cudf::transform_lto(udf,
                                       cudf::lto_binary_type::FATBIN,
                                       cudf::null_aware::YES,
                                       std::nullopt,
                                       inputs,
                                       specs,
                                       {},
                                       std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({2, 3, 4}),
                                 allocated->get_column(0));
  fixed_width_column_wrapper<int32_t> overflow{1, 2, std::numeric_limits<int32_t>::max()};
  inputs[0] = cudf::column_view{overflow};
  EXPECT_THROW(run(), cudf::evaluation_error);
  inputs[0] = cudf::column_view{input};
  run();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({2, 3, 4}), output->view());
}
TEST_F(RowABI, PreparedListAndDictionaryOutputs)
{
  cudf::test::lists_column_wrapper<int32_t> input{{1, 2}, {}, {3}};
  auto output     = cudf::test::lists_column_wrapper<int32_t>{{0, 0}, {}, {0}}.release();
  auto dictionary = cudf::dictionary::encode(fixed_width_column_wrapper<int32_t>{10, 20, 30});
  std::array<cudf::transform_input, 1> inputs{input};
  apply_source(optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto input=cudf_row_input_column(in,0);
      auto output=cudf_row_output_column(out,0);
      for(i32 i=0;i<load_list_size(input,row);++i) {
        auto value=load_list_i32_nullable(input,row,i); value.value*=2;
        auto error=store_optional_list_i32(output,row,i,value);
        if(error!=success) return error;
      }
      return set_dict_i32(out,1,row,(row+1)%3,index_type_tag<i32>{});
    })",
               inputs,
               cudf::mutable_table_view{{output->mutable_view(), dictionary->mutable_view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::test::lists_column_wrapper<int32_t>({{2, 4}, {}, {6}}),
                                 output->view());
  auto expected = cudf::dictionary::encode(fixed_width_column_wrapper<int32_t>{20, 30, 10});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), dictionary->view());
}
TEST_F(RowABI, DictionaryListInputAndPreparedStrings)
{
  auto keys    = cudf::test::lists_column_wrapper<int32_t>{{97, 98}, {99, 100}}.release();
  auto indices = fixed_width_column_wrapper<int8_t>{1, 0}.release();
  auto input   = cudf::make_dictionary_column(std::move(keys), std::move(indices));
  auto output  = cudf::test::strings_column_wrapper{"xx", "xx"}.release();
  std::array<cudf::transform_input, 1> inputs{input->view()};
  apply_source(R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto input=cudf_row_input_column(in,0); auto output=cudf_row_output_column(out,0);
      auto data=string_data(output,row,type_tag<i32>{});
      for(i32 i=0;i<load_dict_list_size(input,row,index_type_tag<i8>{});++i) data[i]=static_cast<char>(load_dict_list_i32(input,row,i,index_type_tag<i8>{})-32);
      return success;
    })",
               inputs,
               cudf::mutable_table_view{{output->mutable_view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::test::strings_column_wrapper({"CD", "AB"}), output->view());
}
TEST_F(RowABI, MinimalHeadersAndFailedCompilationRecovery)
{
  fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  auto output = fixed_width_column_wrapper<int32_t>{0, 0, 0}.release();
  std::array<cudf::transform_input, 1> inputs{input};
  std::string source = optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      return store_optional_i32(out,0,row,load_i32_nullable(in,0,row));
    })";
  EXPECT_THROW(apply_source("#include <cudf/column/column_device_view.cuh>\n" + source,
                            inputs,
                            cudf::mutable_table_view{{output->mutable_view()}}),
               std::exception);
  apply_source(source, inputs, cudf::mutable_table_view{{output->mutable_view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, output->view());
}
TEST_F(RowABI, JoinKindsNullsAndSentinels)
{
  fixed_width_column_wrapper<int32_t> left{{10, 20, 30}, {1, 0, 1}};
  fixed_width_column_wrapper<int32_t> right{5, 25, 40};
  fixed_width_column_wrapper<int32_t> li{0, 0, 1, 2, cudf::JoinNoMatch};
  fixed_width_column_wrapper<int32_t> ri{0, 1, 1, 2, 2};
  auto const left_view  = cudf::column_view{li};
  auto const right_view = cudf::column_view{ri};
  auto const left_map   = cudf::device_span<cudf::size_type const>{
    left_view.data<int32_t>(), static_cast<std::size_t>(left_view.size())};
  auto const right_map = cudf::device_span<cudf::size_type const>{
    right_view.data<int32_t>(), static_cast<std::size_t>(right_view.size())};
  std::string const udf = R"(
    extern "C" __device__ status predicate_operator(void const* in,bool8* outputs,void*,i32 row) {
      auto l=load_i32_nullable(in,0,row); auto r=load_i32_nullable(in,1,row);
      *outputs = l.validity && r.validity && l.value>r.value;
      return success;
    })";
  for (auto kind :
       {cudf::join_kind::INNER_JOIN, cudf::join_kind::LEFT_JOIN, cudf::join_kind::FULL_JOIN}) {
    auto result = cudf::filter_join_indices_jit(
      cudf::table_view{{left}}, cudf::table_view{{right}}, left_map, right_map, udf, kind);
    auto const expected_size = kind == cudf::join_kind::INNER_JOIN  ? 1
                               : kind == cudf::join_kind::LEFT_JOIN ? 3
                                                                    : 5;
    ASSERT_EQ(result.first->size(), expected_size);
    std::vector<int32_t> actual_left(expected_size), actual_right(expected_size);
    CUDF_CUDA_TRY(cudaMemcpy(actual_left.data(),
                             result.first->data(),
                             expected_size * sizeof(int32_t),
                             cudaMemcpyDeviceToHost));
    CUDF_CUDA_TRY(cudaMemcpy(actual_right.data(),
                             result.second->data(),
                             expected_size * sizeof(int32_t),
                             cudaMemcpyDeviceToHost));
    std::vector<std::pair<int32_t, int32_t>> pairs;
    for (int i = 0; i < expected_size; ++i) {
      pairs.emplace_back(actual_left[i], actual_right[i]);
    }
    std::ranges::sort(pairs);
    std::vector<std::pair<int32_t, int32_t>> expected{{0, 0}};
    if (kind != cudf::join_kind::INNER_JOIN) {
      expected.emplace_back(1, cudf::JoinNoMatch);
      expected.emplace_back(2, cudf::JoinNoMatch);
    }
    if (kind == cudf::join_kind::FULL_JOIN) {
      expected.emplace_back(cudf::JoinNoMatch, 1);
      expected.emplace_back(cudf::JoinNoMatch, 2);
    }
    std::ranges::sort(expected);
    EXPECT_EQ(pairs, expected);
  }
  std::string const failing = R"(
    extern "C" __device__ status predicate_operator(void const*,bool8* outputs,void*,i32) { *outputs = true; return arithmetic_overflow; })";
  for (auto kind :
       {cudf::join_kind::INNER_JOIN, cudf::join_kind::LEFT_JOIN, cudf::join_kind::FULL_JOIN}) {
    EXPECT_THROW(
      cudf::filter_join_indices_jit(
        cudf::table_view{{left}}, cudf::table_view{{right}}, left_map, right_map, failing, kind),
      cudf::evaluation_error);
  }
}

TEST_F(RowABI, PredicatePreservesGatheredStringOffsets)
{
  auto left     = cudf::test::strings_column_wrapper{"a", "bb", "ccc"}.release();
  auto contents = left->release();
  contents.children[0] =
    cudf::cast(contents.children[0]->view(), cudf::data_type{cudf::type_id::INT64});
  left = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                        3,
                                        std::move(*contents.data),
                                        std::move(*contents.null_mask),
                                        0,
                                        std::move(contents.children));
  fixed_width_column_wrapper<int32_t> right{3, 1};
  fixed_width_column_wrapper<int32_t> li{2, 0, 1, 2};
  fixed_width_column_wrapper<int32_t> ri{0, 1, 1, 1};
  auto const l             = cudf::column_view{li};
  auto const r             = cudf::column_view{ri};
  std::string const source = R"(
    extern "C" __device__ status predicate_operator(void const* in,bool8* outputs,void*,i32 row) {
      *outputs = load_string_view(in,0,row,type_tag<i64>{}).bytes==load_i32(in,1,row);
      return success;
    })";
  auto result              = cudf::filter_join_indices_jit(cudf::table_view{{left->view()}},
                                              cudf::table_view{{right}},
                                                           {l.data<int32_t>(), 4},
                                                           {r.data<int32_t>(), 4},
                                              source,
                                              cudf::join_kind::INNER_JOIN);
  ASSERT_EQ(result.first->size(), 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fixed_width_column_wrapper<int32_t>({2, 0}),
    cudf::column_view{cudf::data_type{cudf::type_id::INT32}, 2, result.first->data(), nullptr, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fixed_width_column_wrapper<int32_t>({0, 1}),
    cudf::column_view{cudf::data_type{cudf::type_id::INT32}, 2, result.second->data(), nullptr, 0});
}

TEST_F(RowABI, PredicatePreservesGatheredDictionaryIndices)
{
  auto check = []<typename Index>() {
    auto dictionary =
      make_codec_dictionary(fixed_width_column_wrapper<int32_t>{10, 20, 30}.release(),
                            fixed_width_column_wrapper<Index>{2, 0, 1}.release(),
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                            0);
    fixed_width_column_wrapper<int32_t> right{30, 10};
    fixed_width_column_wrapper<int32_t> li{2, 0, 1, 0};
    fixed_width_column_wrapper<int32_t> ri{0, 0, 1, 1};
    auto const l      = cudf::column_view{li};
    auto const r      = cudf::column_view{ri};
    auto const tag    = cudf::jit::row_abi_type(cudf::data_type{cudf::type_to_id<Index>()});
    auto const source = "#define INDEX " + tag + R"(
      extern "C" __device__ status predicate_operator(void const* in,bool8* outputs,void*,i32 row) {
        *outputs = load_dict_i32(in,0,row,index_type_tag<INDEX>{})==load_i32(in,1,row);
        return success;
      })";
    auto result       = cudf::filter_join_indices_jit(cudf::table_view{{dictionary->view()}},
                                                cudf::table_view{{right}},
                                                      {l.data<int32_t>(), 4},
                                                      {r.data<int32_t>(), 4},
                                                source,
                                                cudf::join_kind::INNER_JOIN);
    ASSERT_EQ(result.first->size(), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      fixed_width_column_wrapper<int32_t>({0, 1}),
      cudf::column_view{
        cudf::data_type{cudf::type_id::INT32}, 2, result.first->data(), nullptr, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      fixed_width_column_wrapper<int32_t>({0, 1}),
      cudf::column_view{
        cudf::data_type{cudf::type_id::INT32}, 2, result.second->data(), nullptr, 0});
  };
  check.operator()<int8_t>();
  check.operator()<int16_t>();
  check.operator()<int32_t>();
  check.operator()<int64_t>();
  check.operator()<uint8_t>();
  check.operator()<uint16_t>();
  check.operator()<uint32_t>();
  check.operator()<uint64_t>();
}

TEST_F(RowABI, AstJoinAnsiFailureAndNullification)
{
  fixed_width_column_wrapper<int32_t> left{std::numeric_limits<int32_t>::max()};
  fixed_width_column_wrapper<int32_t> right{1};
  fixed_width_column_wrapper<int32_t> indices{0};
  auto const view = cudf::column_view{indices};
  cudf::device_span<cudf::size_type const> map{view.data<int32_t>(), 1};
  cudf::ast::tree tree;
  cudf::ast::column_reference l{0, cudf::ast::table_reference::LEFT};
  cudf::ast::column_reference r{0, cudf::ast::table_reference::RIGHT};
  auto& add      = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD_OVERFLOW, {l, r});
  auto predicate = cudf::ast::operation(cudf::ast::ast_operator::GREATER, add, r);
  EXPECT_THROW(cudf::filter_join_indices_jit(cudf::table_view{{left}},
                                             cudf::table_view{{right}},
                                             map,
                                             map,
                                             predicate,
                                             cudf::join_kind::INNER_JOIN),
               cudf::evaluation_error);
  auto& null_add = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ADD_OVERFLOW, {l, r}, cudf::error_policy::NULLIFY);
  auto null_predicate = cudf::ast::operation(cudf::ast::ast_operator::GREATER, null_add, r);
  auto result         = cudf::filter_join_indices_jit(cudf::table_view{{left}},
                                              cudf::table_view{{right}},
                                              map,
                                              map,
                                              null_predicate,
                                              cudf::join_kind::INNER_JOIN);
  EXPECT_EQ(result.first->size(), 0);
}

namespace {
rtcx::blob compile_test_fragment(std::string const& name, std::string const& source)
{
  char const* names[]{"cudf/detail/operation_udf.cuh"};
  char const* headers[]{source.c_str()};
  return cudf::get_kernel_fragment(name, "cudf/cpp/src/jit/row_operator.cu", names, headers, name);
}
}  // namespace

TEST_F(RowABI, NamedFragmentContentParticipatesInCacheIdentity)
{
  // Direct kernel calls use the test's statically linked RTCX instance.
  rtcx::initialize();
  auto entry = compile_test_fragment(
    "entry",
    "extern \"C\" __device__ int dependency(); extern \"C\" __global__ void cudf_kernel_entry(int* "
    "out) { *out = dependency(); }");
  cudf::detail::device_scalar<int32_t> output{0, cudf::get_default_stream()};
  for (int value : {1, 5, 1}) {
    auto dependency = compile_test_fragment(
      "same-name",
      "extern \"C\" __device__ int dependency() { return " + std::to_string(value) + "; }");
    rtcx::memory_fragment fragments[]{
      {.data = entry->view(), .type = rtcx::binary_type::LTO_IR, .name = "entry"},
      {.data = dependency->view(), .type = rtcx::binary_type::LTO_IR, .name = "same-name"}};
    auto kernel = cudf::get_lto_linked_kernel("named-fragment-test", {}, fragments);
    kernel.launch_with({1}, {1}, 0, cudf::get_default_stream(), output.data());
    EXPECT_EQ(output.value(cudf::get_default_stream()), value);
  }
  rtcx::teardown();
}

TEST_F(RowABI, EffectfulCallAndUserData)
{
  fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  auto output  = fixed_width_column_wrapper<int32_t>{0, 0, 0}.release();
  auto counter = fixed_width_column_wrapper<int32_t>{0, 0, 0}.release();
  std::array<cudf::transform_input, 1> inputs{input};
  apply_source(optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void* data,i32 row) {
      auto counts=static_cast<i32*>(data); ++counts[row];
      auto value=load_i32_nullable(in,0,row); value.value+=counts[row];
      return store_optional_i32(out,0,row,value);
    })",
               inputs,
               cudf::mutable_table_view{{output->mutable_view()}},
               counter->mutable_view().data<int32_t>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({1, 1, 1}), counter->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({2, 3, 4}), output->view());
}
TEST_F(RowABI, PreparedDecimalPayload)
{
  fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  auto output =
    cudf::test::fixed_point_column_wrapper<int32_t>{{0, 0, 0}, numeric::scale_type{-2}}.release();
  std::array<cudf::transform_input, 1> inputs{input};
  std::string source = optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto value=load_i32_nullable(in,0,row);
      return store_optional_decimal32(out,0,row,{{value.value,SCALE},value.validity});
    })";
  apply_source(
    "#define SCALE -2\n" + source, inputs, cudf::mutable_table_view{{output->mutable_view()}});
  cudf::test::fixed_point_column_wrapper<int32_t> expected{{1, 2, 3}, numeric::scale_type{-2}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}
TEST_F(RowABI, ReflexiveComparisonsPreserveNullAndNaNSemantics)
{
  cudf::test::strings_column_wrapper strings{{"one", "two", ""}, {1, 0, 1}};
  cudf::ast::column_reference reference{0};
  cudf::ast::operation equal{cudf::ast::ast_operator::EQUAL, reference, reference};
  auto result = cudf::compute_column_jit(cudf::table_view{{strings}}, equal);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<bool>({true, false, true}, {1, 0, 1}),
                                 result->view());
  fixed_width_column_wrapper<float> floats{
    std::numeric_limits<float>::quiet_NaN(), 1.0f, std::numeric_limits<float>::quiet_NaN()};
  result = cudf::compute_column_jit(cudf::table_view{{floats}}, equal);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<bool>({false, true, false}),
                                 result->view());
}
TEST_F(RowABI, DictionaryIndexWidths)
{
  std::string identity = optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) { return store_optional_i32(out,0,row,load_i32_nullable(in,0,row)); })";
  auto check           = [&]<typename Index>() {
    auto const tag   = cudf::jit::row_abi_type(cudf::data_type{cudf::type_to_id<Index>()});
    auto source      = identity;
    auto const token = source.find("load_i32_nullable(in,0,row)");
    source.replace(token,
                   std::string{"load_i32_nullable(in,0,row)"}.size(),
                   "optional<i32>{load_dict_i32(in,0,row,index_type_tag<" + tag + ">{ }),1}");
    auto keys    = fixed_width_column_wrapper<int32_t>{10, 20, 30}.release();
    auto indices = fixed_width_column_wrapper<Index>{2, 0, 1}.release();
    auto dictionary =
      make_codec_dictionary(std::move(keys),
                            std::move(indices),
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                            0);
    auto output = fixed_width_column_wrapper<int32_t>{0, 0, 0}.release();
    std::array<cudf::transform_input, 1> inputs{dictionary->view()};
    apply_source(source, inputs, cudf::mutable_table_view{{output->mutable_view()}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({30, 10, 20}),
                                   output->view());
  };
  check.operator()<int8_t>();
  check.operator()<int16_t>();
  check.operator()<int32_t>();
  check.operator()<int64_t>();
  check.operator()<uint8_t>();
  check.operator()<uint16_t>();
  check.operator()<uint32_t>();
  check.operator()<uint64_t>();
}
TEST_F(RowABI, DictionaryListCodecsAndSetters)
{
  auto check = [&]<typename Index>() {
    auto const tag = cudf::jit::row_abi_type(cudf::data_type{cudf::type_to_id<Index>()});
    auto keys = cudf::test::lists_column_wrapper<int32_t>{{10, 11}, {20, 21}, {30, 31}}.release();
    auto indices = fixed_width_column_wrapper<Index>{2, 0, 1}.release();
    auto dictionary =
      make_codec_dictionary(std::move(keys),
                            std::move(indices),
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                            0);
    auto dictionary_output = std::make_unique<cudf::column>(*dictionary);
    auto output            = fixed_width_column_wrapper<int32_t>{0, 0, 0}.release();
    std::array<cudf::transform_input, 1> inputs{dictionary->view()};
    auto source = "#define INDEX " + tag + "\n" + std::string{R"(
      extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
        auto column=cudf_row_input_column(in,0);
        auto value=load_dict_list_i32(column,row,0,index_type_tag<INDEX>{});
        auto error=set_i32(out,0,row,value);
        if(error!=success) return error;
        return set_dict_list_i32(out,1,row,(row+1)%3,index_type_tag<INDEX>{});
      })"};
    apply_source(
      source,
      inputs,
      cudf::mutable_table_view{{output->mutable_view(), dictionary_output->mutable_view()}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({30, 10, 20}),
                                   output->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<Index>({1, 2, 0}),
                                   dictionary_output->child(0).view());
    // RowIR selects the same physical index overload for scalar dictionary keys.
    auto scalar_keys    = fixed_width_column_wrapper<int32_t>{10, 20, 30}.release();
    auto scalar_indices = fixed_width_column_wrapper<Index>{2, 0, 1}.release();
    auto scalar_dict =
      make_codec_dictionary(std::move(scalar_keys),
                            std::move(scalar_indices),
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                            0);
    auto ref        = cudf::ast::column_reference{0};
    auto ast_output = cudf::compute_column_jit(cudf::table_view{{scalar_dict->view()}}, ref);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({30, 10, 20}),
                                   ast_output->view());
  };
  check.operator()<int8_t>();
  check.operator()<int16_t>();
  check.operator()<int32_t>();
  check.operator()<int64_t>();
  check.operator()<uint8_t>();
  check.operator()<uint16_t>();
  check.operator()<uint32_t>();
  check.operator()<uint64_t>();
}

TEST_F(RowABI, String64BitOffsetsAndNullableListChildren)
{
  auto make_large = [](std::unique_ptr<cudf::column> column) {
    auto const rows = column->size();
    auto contents   = column->release();
    contents.children[0] =
      cudf::cast(contents.children[0]->view(), cudf::data_type{cudf::type_id::INT64});
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                          rows,
                                          std::move(*contents.data),
                                          std::move(*contents.null_mask),
                                          0,
                                          std::move(contents.children));
  };
  auto strings     = make_large(cudf::test::strings_column_wrapper{"ab", "cd", ""}.release());
  auto output      = make_large(cudf::test::strings_column_wrapper{"xx", "xx", ""}.release());
  std::string copy = R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto input=load_string_view_nullable(in,0,row,type_tag<i64>{}); auto column=cudf_row_output_column(out,0);
      auto data=string_data(column,row,type_tag<i64>{});
      for(i32 i=0;i<input.value.bytes;++i) data[i]=input.value.data[i];
      return success;
    })";
  std::array<cudf::transform_input, 1> inputs{strings->view()};
  apply_source(copy, inputs, cudf::mutable_table_view{{output->mutable_view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings->view(), output->view());
  std::string typed_copy = optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      return store_optional_string_view(out,0,row,load_string_view_nullable(in,0,row,type_tag<i64>{}),type_tag<i64>{});
    })";
  apply_source(typed_copy, inputs, cudf::mutable_table_view{{output->mutable_view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings->view(), output->view());
  auto make_list = [](std::unique_ptr<cudf::column> child) {
    return cudf::make_lists_column(3,
                                   fixed_width_column_wrapper<int32_t>{0, 2, 2, 3}.release(),
                                   std::move(child),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  };
  auto lists       = make_list(fixed_width_column_wrapper<int32_t>({1, 2, 3}, {1, 0, 1}).release());
  auto list_output = make_list(fixed_width_column_wrapper<int32_t>({0, 0, 0}, {1, 1, 1}).release());
  std::string list_copy = optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto input=cudf_row_input_column(in,0); auto output=cudf_row_output_column(out,0);
      for(i32 i=0;i<load_list_size(input,row);++i) {
        auto error=store_optional_list_i32(output,row,i,load_list_i32_nullable(input,row,i));
        if(error!=success) return error;
      }
      return success;
    })";
  inputs[0]             = lists->view();
  apply_source(list_copy, inputs, cudf::mutable_table_view{{list_output->mutable_view()}});
  list_output->child(1).set_null_count(1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(lists->view(), list_output->view());
  auto string_lists =
    make_list(cudf::test::strings_column_wrapper({"ab", "cd", "ef"}, {1, 0, 1}).release());
  auto string_output =
    make_list(cudf::test::strings_column_wrapper({"xx", "xx", "xx"}, {1, 1, 1}).release());
  inputs[0]                    = string_lists->view();
  std::string string_list_copy = optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto input=cudf_row_input_column(in,0); auto output=cudf_row_output_column(out,0);
      for(i32 i=0;i<load_list_size(input,row);++i) {
        auto error=store_optional_list_string_view(output,row,i,load_list_string_view_nullable(input,row,i,type_tag<i32>{}),type_tag<i32>{});
        if(error!=success) return error;
      }
      return success;
    })";
  apply_source(string_list_copy, inputs, cudf::mutable_table_view{{string_output->mutable_view()}});
  string_output->child(1).set_null_count(1);
  // Prepared offsets are immutable. Normalize retained character regions after clearing validity.
  auto normalized = cudf::purge_nonempty_nulls(string_output->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(string_lists->view(), normalized->view());
}

TEST_F(RowABI, DictionaryStringOffsetAndIndexCodecs)
{
  auto strings = [](bool large, bool empty_values) {
    auto column = empty_values ? cudf::test::strings_column_wrapper{"xx", "xx", "xx"}.release()
                               : cudf::test::strings_column_wrapper{"ab", "cd", "ef"}.release();
    if (!large) return column;
    auto contents = column->release();
    contents.children[0] =
      cudf::cast(contents.children[0]->view(), cudf::data_type{cudf::type_id::INT64});
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                          3,
                                          std::move(*contents.data),
                                          std::move(*contents.null_mask),
                                          0,
                                          std::move(contents.children));
  };
  auto lists = [](std::unique_ptr<cudf::column> child) {
    return cudf::make_lists_column(3,
                                   fixed_width_column_wrapper<int32_t>{0, 1, 2, 3}.release(),
                                   std::move(child),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  };
  auto check = [&]<typename Index>() {
    auto const index_tag = cudf::jit::row_abi_type(cudf::data_type{cudf::type_to_id<Index>()});
    for (bool large : {false, true}) {
      auto dictionary =
        make_codec_dictionary(strings(large, false),
                              fixed_width_column_wrapper<Index>{2, 0, 1}.release(),
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                              0);
      auto dictionary_list =
        make_codec_dictionary(lists(strings(large, false)),
                              fixed_width_column_wrapper<Index>{2, 0, 1}.release(),
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                              0);
      auto output                 = strings(large, true);
      auto list_output            = lists(strings(large, true));
      auto dictionary_output      = std::make_unique<cudf::column>(*dictionary);
      auto dictionary_list_output = std::make_unique<cudf::column>(*dictionary_list);
      std::array<cudf::transform_input, 2> inputs{dictionary->view(), dictionary_list->view()};
      auto source = "#define INDEX " + index_tag + "\n#define OFFSET " + (large ? "i64" : "i32") +
                    "\n" + std::string{R"(
        extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
          auto list=cudf_row_input_column(in,1);
          auto a=load_dict_string_view(in,0,row,index_type_tag<INDEX>{},type_tag<OFFSET>{});
          auto b=load_dict_list_string_view(list,row,0,index_type_tag<INDEX>{},type_tag<OFFSET>{});
          auto error=set_string_view(out,0,row,a,type_tag<OFFSET>{});
          if(error!=success) return error;
          error=set_list_string_view(cudf_row_output_column(out,1),row,0,b,type_tag<OFFSET>{});
          if(error!=success) return error;
          error=set_dict_string_view(out,2,row,(row+1)%3,index_type_tag<INDEX>{},type_tag<OFFSET>{});
          if(error!=success) return error;
          return set_dict_list_string_view(out,3,row,(row+1)%3,index_type_tag<INDEX>{},type_tag<OFFSET>{});
        })"};
      apply_source(source,
                   inputs,
                   cudf::mutable_table_view{{output->mutable_view(),
                                             list_output->mutable_view(),
                                             dictionary_output->mutable_view(),
                                             dictionary_list_output->mutable_view()}});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::test::strings_column_wrapper({"ef", "ab", "cd"}),
                                     output->view());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::test::strings_column_wrapper({"ef", "ab", "cd"}),
                                     list_output->child(1).view());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<Index>({1, 2, 0}),
                                     dictionary_output->child(0).view());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<Index>({1, 2, 0}),
                                     dictionary_list_output->child(0).view());
      auto ref        = cudf::ast::column_reference{0};
      auto ast_output = cudf::compute_column_jit(cudf::table_view{{dictionary->view()}}, ref);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::test::strings_column_wrapper({"ef", "ab", "cd"}),
                                     ast_output->view());
    }
  };
  check.operator()<int8_t>();
  check.operator()<int16_t>();
  check.operator()<int32_t>();
  check.operator()<int64_t>();
  check.operator()<uint8_t>();
  check.operator()<uint16_t>();
  check.operator()<uint32_t>();
  check.operator()<uint64_t>();
}

TEST_F(RowABI, ExistingAllocatedApiDefaultsToRowMode)
{
  fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  std::array<cudf::transform_input, 1> inputs{input};
  std::array<cudf::transform_output, 1> outputs{
    {{cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::ALL_VALID}}};
  std::string source = optional_stores + R"(
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto value=load_i32_nullable(in,0,row); value.value+=7;
      return store_optional_i32(out,0,row,value);
    })";
  auto result        = cudf::transform(source,
                                cudf::udf_source_type::CUDA,
                                cudf::null_aware::YES,
                                std::nullopt,
                                inputs,
                                outputs,
                                       {},
                                std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({8, 9, 10}),
                                 result->get_column(0));
  cudf::transform_program program{
    source, cudf::udf_source_type::CUDA, cudf::null_aware::YES, std::nullopt, inputs, outputs, {}};
  result = program.run(inputs, outputs, {}, std::nullopt);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({8, 9, 10}),
                                 result->get_column(0));
}
TEST_F(RowABI, PreparedCudaValueUdfUsesRowAdapter)
{
  fixed_width_column_wrapper<int32_t> input{{1, 2, 3}, {1, 0, 1}};
  auto output = fixed_width_column_wrapper<int32_t>{{0, 0, 0}, {0, 1, 0}}.release();
  std::array<cudf::transform_input, 1> inputs{input};
  cudf::transform("__device__ int op(int* out,int in) { *out=in+2; return 0; }",
                  cudf::udf_source_type::CUDA,
                  cudf::null_aware::NO,
                  std::nullopt,
                  inputs,
                  cudf::mutable_table_view{{output->mutable_view()}},
                  cudf::get_default_stream(),
                  cudf::get_current_device_resource_ref(),
                  false);
  output->set_null_count(1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({3, 4, 5}, {1, 0, 1}),
                                 output->view());
}

TEST_F(RowABI, TransformAndPredicateLinkHelperDependencies)
{
  rtcx::initialize();
  auto helper = compile_test_fragment(
    "helper", "extern \"C\" __device__ int helper(int value) { return value + 1; }");
  std::array<cudf::lto_fragment, 1> dependencies{{{helper->view(), cudf::lto_binary_type::LTO_IR}}};
  fixed_width_column_wrapper<int32_t> input{1, 2, 3};
  auto output = fixed_width_column_wrapper<int32_t>{0, 0, 0}.release();
  std::array<cudf::transform_input, 1> inputs{input};
  cudf::transform(optional_stores + R"(
    extern "C" __device__ i32 helper(i32);
    extern "C" __device__ status transform_operator(void const* in,void const* out,void*,i32 row) {
      auto value=load_i32_nullable(in,0,row);
      value.value=helper(value.value);
      return store_optional_i32(out,0,row,value);
    })",
                  cudf::udf_source_type::CUDA,
                  cudf::null_aware::YES,
                  std::nullopt,
                  inputs,
                  cudf::mutable_table_view{{output->mutable_view()}},
                  cudf::get_default_stream(),
                  cudf::get_current_device_resource_ref(),
                  true,
                  dependencies);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fixed_width_column_wrapper<int32_t>({2, 3, 4}), output->view());
  fixed_width_column_wrapper<int32_t> map{0, 1, 2};
  auto map_view = cudf::column_view{map};
  auto indices  = cudf::device_span<cudf::size_type const>{map_view.data<int32_t>(), 3};
  auto result   = cudf::filter_join_indices_jit(cudf::table_view{{input}},
                                              cudf::table_view{{input}},
                                              indices,
                                              indices,
                                              R"(
    extern "C" __device__ i32 helper(i32);
    extern "C" __device__ status predicate_operator(void const* in,bool8* outputs,void*,i32 row) {
      auto l=load_i32_nullable(in,0,row); auto r=load_i32_nullable(in,1,row);
      *outputs = helper(l.value)>r.value;
      return success;
    })",
                                              cudf::join_kind::INNER_JOIN,
                                              false,
                                              cudf::get_default_stream(),
                                              cudf::get_current_device_resource_ref(),
                                              true,
                                              std::nullopt,
                                              dependencies);
  EXPECT_EQ(result.first->size(), 3);
  rtcx::teardown();
}

TEST_F(RowABI, AstProgramRejectsChangedStringOffsetWidth)
{
  auto input = cudf::test::strings_column_wrapper({"ab", "cd"}).release();
  auto ref   = cudf::ast::column_reference{0};
  std::array<std::reference_wrapper<cudf::ast::expression const>, 1> expressions{ref};
  cudf::transform_program program{cudf::table_view{{*input}}, expressions};
  auto result = program.run(cudf::table_view{{*input}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input->view(), result->get_column(0).view());

  auto large    = std::make_unique<cudf::column>(input->view());
  auto contents = large->release();
  contents.children[0] =
    cudf::cast(contents.children[0]->view(), cudf::data_type{cudf::type_id::INT64});
  large = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                         input->size(),
                                         std::move(*contents.data),
                                         std::move(*contents.null_mask),
                                         0,
                                         std::move(contents.children));
  EXPECT_THROW(program.run(cudf::table_view{{*large}}), std::invalid_argument);
}
