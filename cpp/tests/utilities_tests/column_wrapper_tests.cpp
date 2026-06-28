/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/iterator>

namespace {

template <typename Factory>
void expect_output_uses_resource(Factory factory)
{
  auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  {
    auto column = factory(mr).release();
    cudf::test::get_default_stream().synchronize();
    auto const bytes = mr.get_bytes_counter();
    EXPECT_EQ(column->alloc_size(), static_cast<std::size_t>(bytes.value));
    EXPECT_EQ(column->alloc_size(), static_cast<std::size_t>(bytes.total));
  }

  cudf::test::get_default_stream().synchronize();
  EXPECT_EQ(0, mr.get_bytes_counter().value);
}

template <typename Factory>
void expect_output_uses_distinct_resources(Factory factory)
{
  auto upstream     = cudf::get_current_device_resource_ref();
  auto output_mr    = rmm::mr::statistics_resource_adaptor(upstream);
  auto temporary_mr = rmm::mr::statistics_resource_adaptor(upstream);

  {
    auto column = factory(cudf::memory_resources{output_mr, temporary_mr}).release();
    cudf::test::get_default_stream().synchronize();
    auto const output_bytes    = output_mr.get_bytes_counter();
    auto const temporary_bytes = temporary_mr.get_bytes_counter();
    EXPECT_EQ(column->alloc_size(), static_cast<std::size_t>(output_bytes.value));
    EXPECT_EQ(column->alloc_size(), static_cast<std::size_t>(output_bytes.total));
    EXPECT_EQ(0, temporary_bytes.value);
    EXPECT_EQ(0, temporary_bytes.total);
  }

  cudf::test::get_default_stream().synchronize();
  EXPECT_EQ(0, output_mr.get_bytes_counter().value);
  EXPECT_EQ(0, temporary_mr.get_bytes_counter().value);
}

}  // namespace

TEST(FixedWidthColumnWrapperMemoryResourceTest, ExplicitResourceOverloadMatrix)
{
  auto const elements = std::vector<int32_t>{1, 2, 3, 4};
  auto const validity = std::vector<bool>{true, false, true, false};

  expect_output_uses_resource(
    [](auto& mr) { return cudf::test::fixed_width_column_wrapper<int32_t>(mr); });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_width_column_wrapper<int32_t>(elements.begin(), elements.end(), mr);
  });

  expect_output_uses_resource([](auto& mr) {
    auto ref = rmm::device_async_resource_ref{mr};
    return cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4}, ref);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_width_column_wrapper<int32_t>(
      elements.begin(), elements.end(), validity.begin(), mr);
  });

  expect_output_uses_resource([](auto& mr) {
    return cudf::test::fixed_width_column_wrapper<int32_t>(
      {1, 2, 3, 4}, {true, false, true, false}, mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4}, validity.begin(), mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_width_column_wrapper<int32_t>(
      elements.begin(), elements.end(), {true, false, true, false}, mr);
  });

  expect_output_uses_resource([](auto& mr) {
    using pair_type = std::pair<int32_t, bool>;
    return cudf::test::fixed_width_column_wrapper<int32_t>(
      {pair_type{1, true}, pair_type{2, false}, pair_type{3, true}}, mr);
  });
}

TEST(FixedWidthColumnWrapperMemoryResourceTest, DistinctOutputAndTemporaryResources)
{
  auto const elements = std::vector<int32_t>{1, 2, 3, 4};
  auto const validity = std::vector<bool>{true, false, true, false};

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::fixed_width_column_wrapper<int32_t>(
      elements.begin(), elements.end(), validity.begin(), mr);
  });
}

TEST(FixedWidthColumnWrapperMemoryResourceTest, FixedPointElementPaths)
{
  using decimal32 = numeric::decimal32;

  auto const reps     = std::vector<int32_t>{1, 2, 3, 4};
  auto const decimals = std::vector<decimal32>{decimal32{1, numeric::scale_type{0}},
                                               decimal32{2, numeric::scale_type{0}},
                                               decimal32{3, numeric::scale_type{0}},
                                               decimal32{4, numeric::scale_type{0}}};

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_width_column_wrapper<decimal32, int32_t>(reps.begin(), reps.end(), mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_width_column_wrapper<decimal32>(decimals.begin(), decimals.end(), mr);
  });
}

TEST(FixedPointColumnWrapperMemoryResourceTest, ExplicitResourceOverloadMatrix)
{
  auto const elements = std::vector<int32_t>{1, 2, 3, 4};
  auto const validity = std::vector<bool>{true, false, true, false};
  auto const scale    = numeric::scale_type{-2};

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      elements.begin(), elements.end(), scale, mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    auto ref = rmm::device_async_resource_ref{mr};
    return cudf::test::fixed_point_column_wrapper<int32_t>({1, 2, 3, 4}, scale, ref);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      elements.begin(), elements.end(), validity.begin(), scale, mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      {1, 2, 3, 4}, {true, false, true, false}, scale, mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      {1, 2, 3, 4}, validity.begin(), scale, mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      elements.begin(), elements.end(), {true, false, true, false}, scale, mr);
  });
}

TEST(FixedPointColumnWrapperMemoryResourceTest, DistinctOutputAndTemporaryResources)
{
  auto const elements = std::vector<int32_t>{1, 2, 3, 4};
  auto const validity = std::vector<bool>{true, false, true, false};
  auto const scale    = numeric::scale_type{-2};

  expect_output_uses_distinct_resources([&](auto mr) {
    return cudf::test::fixed_point_column_wrapper<int32_t>(
      elements.begin(), elements.end(), validity.begin(), scale, mr);
  });
}

TEST(StringsColumnWrapperMemoryResourceTest, ExplicitResourceOverloadMatrix)
{
  auto const strings  = std::vector<std::string>{"", "alpha", "beta", "gamma"};
  auto const validity = std::vector<bool>{true, false, true, false};

  expect_output_uses_resource([](auto& mr) { return cudf::test::strings_column_wrapper(mr); });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), mr);
  });

  expect_output_uses_resource([](auto& mr) {
    auto ref = rmm::device_async_resource_ref{mr};
    return cudf::test::strings_column_wrapper({"", "alpha", "beta", "gamma"}, ref);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), validity.begin(), mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::strings_column_wrapper({"", "alpha", "beta", "gamma"}, validity.begin(), mr);
  });

  expect_output_uses_resource([](auto& mr) {
    return cudf::test::strings_column_wrapper(
      {"", "alpha", "beta", "gamma"}, {true, false, true, false}, mr);
  });

  expect_output_uses_resource([](auto& mr) {
    using pair_type = std::pair<std::string, bool>;
    return cudf::test::strings_column_wrapper(
      {pair_type{"", true}, pair_type{"alpha", false}, pair_type{"beta", true}}, mr);
  });
}

TEST(DictionaryColumnWrapperMemoryResourceTest, FixedWidthExplicitResourceOverloadMatrix)
{
  auto const elements = std::vector<int32_t>{3, 1, 3, 2};
  auto const validity = std::vector<bool>{true, false, true, true};

  expect_output_uses_resource(
    [](auto& mr) { return cudf::test::dictionary_column_wrapper<int32_t>(mr); });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::dictionary_column_wrapper<int32_t>(elements.begin(), elements.end(), mr);
  });

  expect_output_uses_resource([](auto& mr) {
    auto ref = rmm::device_async_resource_ref{mr};
    return cudf::test::dictionary_column_wrapper<int32_t>({3, 1, 3, 2}, ref);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::dictionary_column_wrapper<int32_t>(
      elements.begin(), elements.end(), validity.begin(), mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::dictionary_column_wrapper<int32_t>({3, 1, 3, 2}, validity.begin(), mr);
  });

  expect_output_uses_resource([](auto& mr) {
    return cudf::test::dictionary_column_wrapper<int32_t>(
      {3, 1, 3, 2}, {true, false, true, true}, mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::dictionary_column_wrapper<int32_t>(
      elements.begin(), elements.end(), {true, false, true, true}, mr);
  });
}

TEST(DictionaryColumnWrapperMemoryResourceTest, StringExplicitResourceOverloadMatrix)
{
  auto const strings  = std::vector<std::string>{"gamma", "alpha", "gamma", "beta"};
  auto const validity = std::vector<bool>{true, false, true, true};

  expect_output_uses_resource(
    [](auto& mr) { return cudf::test::dictionary_column_wrapper<std::string>(mr); });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::dictionary_column_wrapper<std::string>(strings.begin(), strings.end(), mr);
  });

  expect_output_uses_resource([](auto& mr) {
    auto ref = rmm::device_async_resource_ref{mr};
    return cudf::test::dictionary_column_wrapper<std::string>({"gamma", "alpha", "gamma", "beta"},
                                                              ref);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::dictionary_column_wrapper<std::string>(
      strings.begin(), strings.end(), validity.begin(), mr);
  });

  expect_output_uses_resource([&](auto& mr) {
    return cudf::test::dictionary_column_wrapper<std::string>(
      {"gamma", "alpha", "gamma", "beta"}, validity.begin(), mr);
  });

  expect_output_uses_resource([](auto& mr) {
    return cudf::test::dictionary_column_wrapper<std::string>(
      {"gamma", "alpha", "gamma", "beta"}, {true, false, true, true}, mr);
  });
}

TEST(DictionaryColumnWrapperMemoryResourceTest, EmptyStringDictionaryPreservesChildTypes)
{
  auto mr         = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());
  auto column     = cudf::test::dictionary_column_wrapper<std::string>(mr).release();
  auto dictionary = cudf::dictionary_column_view{column->view()};

  EXPECT_EQ(0, column->size());
  EXPECT_EQ(cudf::type_id::STRING, dictionary.keys().type().id());
  EXPECT_EQ(cudf::type_id::INT32, dictionary.indices().type().id());
}

template <typename T>
struct FixedWidthColumnWrapperTest : public cudf::test::BaseFixture,
                                     cudf::test::UniformRandomGenerator<cudf::size_type> {
  FixedWidthColumnWrapperTest() : cudf::test::UniformRandomGenerator<cudf::size_type>{1000, 5000} {}

  auto size() { return this->generate(); }

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(FixedWidthColumnWrapperTest, cudf::test::FixedWidthTypes);

TYPED_TEST(FixedWidthColumnWrapperTest, EmptyIterator)
{
  auto sequence = cuda::counting_iterator{0};
  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}
TYPED_TEST(FixedWidthColumnWrapperTest, EmptyList)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col{};
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableIteratorConstructor)
{
  auto sequence = cuda::counting_iterator{0};

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NonNullableListConstructor)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5});

  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllValid)
{
  auto sequence = cuda::counting_iterator{0};

  auto all_valid = cudf::test::iterators::no_nulls();

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, all_valid);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllValid)
{
  auto all_valid = cudf::test::iterators::no_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_valid);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableIteratorConstructorAllNull)
{
  auto sequence = cuda::counting_iterator{0};

  auto all_null = cudf::test::iterators::all_nulls();

  auto size = this->size();

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + size, all_null);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), size);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), size);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullableListConstructorAllNull)
{
  auto all_null = cudf::test::iterators::all_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_null);
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullablePairListConstructorAllNull)
{
  using p = std::pair<int32_t, bool>;
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col(
    {p{1, false}, p{2, false}, p{3, false}, p{4, false}, p{5, false}});
  cudf::column_view view = col;

  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, NullablePairListConstructorAllNullMatch)
{
  auto odd_valid = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> match_col({1, 2, 3, 4, 5}, odd_valid);
  cudf::column_view match_view = match_col;

  using p = std::pair<int32_t, bool>;
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({p{1, odd_valid[0]},
                                                                  p{2, odd_valid[1]},
                                                                  p{3, odd_valid[2]},
                                                                  p{4, odd_valid[3]},
                                                                  p{5, odd_valid[4]}});
  cudf::column_view view = col;

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view, match_view);
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllValid)
{
  auto all_valid = cudf::test::iterators::no_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_valid);
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(FixedWidthColumnWrapperTest, ReleaseWrapperAllNull)
{
  auto all_null = cudf::test::iterators::all_nulls();

  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> col({1, 2, 3, 4, 5}, all_null);
  auto colPtr            = col.release();
  cudf::column_view view = *colPtr;
  EXPECT_EQ(view.size(), 5);
  EXPECT_NE(nullptr, view.head());
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_TRUE(view.nullable());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
  EXPECT_EQ(view.offset(), 0);
}

template <typename T>
struct StringsColumnWrapperTest : public cudf::test::BaseFixture,
                                  cudf::test::UniformRandomGenerator<cudf::size_type> {
  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(StringsColumnWrapperTest, cudf::test::StringTypes);

TYPED_TEST(StringsColumnWrapperTest, EmptyList)
{
  cudf::test::strings_column_wrapper col;
  cudf::column_view view = col;
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.head(), nullptr);
  EXPECT_EQ(view.type(), this->data_type());
  EXPECT_FALSE(view.nullable());
  EXPECT_FALSE(view.has_nulls());
  EXPECT_EQ(view.offset(), 0);
}

TYPED_TEST(StringsColumnWrapperTest, NullablePairListConstructorAllNull)
{
  using p = std::pair<std::string, bool>;
  cudf::test::strings_column_wrapper col(
    {p{"a", false}, p{"string", false}, p{"test", false}, p{"for", false}, p{"nulls", false}});
  cudf::strings_column_view view = cudf::column_view(col);

  constexpr auto count = 5;
  EXPECT_EQ(view.size(), count);
  EXPECT_EQ(view.offsets().size(), count + 1);
  // all null entries results in no data allocated to chars
  EXPECT_EQ(nullptr, view.parent().head());
  EXPECT_NE(nullptr, view.offsets().head());
  EXPECT_TRUE(view.has_nulls());
  EXPECT_EQ(view.null_count(), 5);
}

TYPED_TEST(StringsColumnWrapperTest, NullablePairListConstructorAllNullMatch)
{
  auto odd_valid = cudf::test::iterators::nulls_at_multiples_of(2);

  cudf::test::strings_column_wrapper match_col({"a", "string", "", "test", "for", "nulls"},
                                               odd_valid);
  cudf::column_view match_view = match_col;

  using p = std::pair<std::string, bool>;
  cudf::test::strings_column_wrapper col({p{"a", odd_valid[0]},
                                          p{"string", odd_valid[1]},
                                          p{"", odd_valid[2]},
                                          p{"test", odd_valid[3]},
                                          p{"for", odd_valid[4]},
                                          p{"nulls", odd_valid[5]}});
  cudf::column_view view = col;

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view, match_view);
}
