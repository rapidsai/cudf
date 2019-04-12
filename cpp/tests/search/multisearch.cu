/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#include "multisearch.cuh"

#include <random>

using cudf::test::column_wrapper;
using std::get;
using std::cbegin;
using std::cend;

// TODO: Make this templated on a tuple type, and use the gadget above to be able to work with the parameter packs directly.
struct Multisearch : public GdfTest {
    std::default_random_engine randomness_generator;
    std::uniform_int_distribution<gdf_size_type> column_size_distribution{1000, 10000};
    gdf_size_type random_column_size() { return column_size_distribution(randomness_generator); }

    Multisearch() = default;
    ~Multisearch() = default;
};


TEST(Multisearch, fails_when_no_haystack_columns_provided)
{
    gdf_size_type num_columns { 0 };
    gdf_column results;
    gdf_column single_haystack_column;
    gdf_column* haystack_columns[] = { & single_haystack_column };
    gdf_column single_needle_column;
    gdf_column* needle_columns[] = { & single_needle_column };

    ASSERT_THROW(
        gdf_multisearch(
            &results,
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater,
            nulls_appear_before_values,
            use_haystack_length_for_not_found),
        std::invalid_argument
    );
}

TEST(Multisearch, single_non_null_column_one_needle)
{
    using single_element_type = int32_t;

    gdf_column_index_type num_columns     { 1 };

    // single_element_type uniform_value { 123 };
    std::vector<single_element_type> haystack_data { 10, 20, 30, 40, 50 };
    std::vector<single_element_type> needle_data { 20 };
    std::vector<gdf_size_type> dummy_result_data { 1234567 };

    auto single_haystack_column = column_wrapper<single_element_type>(haystack_data);
    auto single_needle_column   = column_wrapper<single_element_type>(needle_data);
//  auto results                = column_wrapper<gdf_size_type      >(needle_data.size(), non_nullable);
    auto results                = column_wrapper<gdf_size_type>(dummy_result_data);

    gdf_column* haystack_columns[] = { single_haystack_column.get() };
    gdf_column* needle_columns[]   = { single_needle_column.get()   };

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater,
            nulls_appear_before_values,
            use_haystack_length_for_not_found)
    );
    auto results_on_host = results.to_host();
    ASSERT_EQ(results.get()->valid, nullptr); // Just a sanity check really
    ASSERT_EQ(get<0>(results_on_host).size(), needle_data.size());
    ASSERT_EQ(get<0>(results_on_host)[0], 2); // at position 2 we have 30, greater than 20.
}

TEST(Multisearch, single_non_null_column_multiple_needles)
{
    using single_element_type = int32_t;

    gdf_column_index_type num_columns     { 1 };

    std::vector<single_element_type> haystack_data       { 10, 20, 30, 40, 50 };
    std::vector<single_element_type> needle_data         {  0,  7, 10, 11, 30, 32, 40, 47, 50, 90 };
    std::vector<gdf_size_type> first_greater_than        {  0,  0,  1,  1,  3,  3,  4,  4,  5,  5 };
    std::vector<gdf_size_type> first_greater_or_equal_to {  0,  0,  0,  1,  2,  3,  3,  4,  4,  5 };

    auto single_haystack_column = column_wrapper<single_element_type>(haystack_data);
    auto single_needle_column   = column_wrapper<single_element_type>(needle_data);
    auto results                = column_wrapper<gdf_size_type      >(needle_data.size(), non_nullable);

    gdf_column* haystack_columns[] = { single_haystack_column.get() };
    gdf_column* needle_columns[]   = { single_needle_column.get()   };

    ASSERT_CUDF_SUCCESS( fill(*results.get(), gdf_size_type{0xDEADBEEF}) );

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater,
            nulls_appear_before_values,
            use_haystack_length_for_not_found)
    );

    auto results_on_host = results.to_host();
    ASSERT_EQ(results.get()->null_count, 0);
    ASSERT_EQ(get<0>(results_on_host).size(), needle_data.size());
    auto results_data_on_host = get<0>(results_on_host);

    ASSERT_EQ(
        std::equal(cbegin(results_data_on_host), cbegin(results_data_on_host), cbegin(first_greater_than) ),
        true);

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater_or_equal,
            nulls_appear_before_values,
            use_haystack_length_for_not_found)
    );

    results_on_host = results.to_host();
    ASSERT_EQ(results.get()->null_count, 0);
    ASSERT_EQ(get<0>(results_on_host).size(), needle_data.size());
    results_data_on_host = get<0>(results_on_host);

    EXPECT_TRUE(std::equal(cbegin(results_data_on_host), cbegin(results_data_on_host), cbegin(first_greater_or_equal_to) ));
}

TEST(Multisearch, single_column_multiple_needles_nulls__find_greater__before_other_values)
{
    const auto print_all_unequal_pairs { false };
    using single_element_type = int32_t;
    gdf_column_index_type num_columns     { 1 };

    std::vector<single_element_type> haystack_data     { 10, 60, 10, 20, 30, 40, 50 };
    std::vector<gdf_valid_type     > haystack_validity {  0,  0,  1,  1,  1,  1,  1 };
    std::vector<single_element_type> needle_data       {  8,  8, 10, 11, 30, 32, 40, 47, 50, 90 };
    std::vector<single_element_type> needle_validity   {  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 };
    std::vector<gdf_size_type> first_greater_than_data {  2,  2,  3,  3,  5,  5,  6,  6,  7,  7 };

    auto single_haystack_column = column_wrapper<single_element_type>(haystack_data, make_validity_initializer(haystack_validity));
    auto single_needle_column   = column_wrapper<single_element_type>(needle_data, make_validity_initializer(needle_validity));
    auto results                = column_wrapper<gdf_size_type      >(needle_data.size(), non_nullable);

//    self_titled_print(single_haystack_column);
//    self_titled_print(single_needle_column);

    gdf_column* haystack_columns[] = { single_haystack_column.get() };
    gdf_column* needle_columns[]   = { single_needle_column.get()   };

    ASSERT_CUDF_SUCCESS(fill(*results.get(), single_element_type{0xDEADBEEF}));

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater,
            nulls_appear_before_values,
            use_haystack_length_for_not_found)
    );

    auto first_greater_than = column_wrapper<gdf_size_type>(first_greater_than_data);

//    print(results, "actual results");
//    print(first_greater_than, "expected results");

    expect_column(results, first_greater_than, print_all_unequal_pairs);
}

TEST(Multisearch, single_column_multiple_needles_nulls__find_greater_or_equal__before_other_values)
{
    const auto print_all_unequal_pairs { false };
    using single_element_type = int32_t;
    gdf_column_index_type num_columns     { 1 };

    std::vector<single_element_type> haystack_data     { 10, 60, 10, 20, 30, 40, 50 };
    std::vector<gdf_valid_type     > haystack_validity {  0,  0,  1,  1,  1,  1,  1 };
    std::vector<single_element_type> needle_data       {  8,  8, 10, 11, 30, 32, 40, 47, 50, 90 };
    std::vector<single_element_type> needle_validity   {  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 };
    std::vector<gdf_size_type> first_greater_or_equal_to_data
                                                       {  0,  2,  2,  3,  4,  5,  5,  6,  6,  7 };

    auto single_haystack_column = column_wrapper<single_element_type>(haystack_data, make_validity_initializer(haystack_validity));
    auto single_needle_column   = column_wrapper<single_element_type>(needle_data, make_validity_initializer(needle_validity));
    auto results                = column_wrapper<gdf_size_type      >(needle_data.size(), non_nullable);

//    self_titled_print(single_haystack_column);
//    self_titled_print(single_needle_column);

    gdf_column* haystack_columns[] = { single_haystack_column.get() };
    gdf_column* needle_columns[]   = { single_needle_column.get()   };

    ASSERT_CUDF_SUCCESS(fill(*results.get(), single_element_type{0xDEADBEEF}));

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater_or_equal,
            nulls_appear_before_values,
            use_haystack_length_for_not_found)
    );

    cudaDeviceSynchronize();

    auto first_greater_or_equal_to = column_wrapper<gdf_size_type>(first_greater_or_equal_to_data);

//    print(results, "actual results");
//    print(first_greater_or_equal_to, "expected results");

    expect_column(results, first_greater_or_equal_to, print_all_unequal_pairs);
}

TEST(Multisearch, single_column_multiple_needles__find_first_greater__nulls_after_other_values)
{
    const auto print_all_unequal_pairs { false };
    using single_element_type = int32_t;
    gdf_column_index_type num_columns     { 1 };

    std::vector<single_element_type> haystack_data     { 10, 20, 30, 40, 50, 10, 60 };
    std::vector<gdf_valid_type     > haystack_validity {  1,  1,  1,  1,  1,  0,  0 };
    std::vector<single_element_type> needle_data       {  8, 10, 11, 30, 32, 40, 47, 50, 90,  8 };
    std::vector<single_element_type> needle_validity   {  1,  1,  1,  1,  1,  1,  1,  1,  1,  0 };
    std::vector<gdf_size_type> first_greater_than_data {  0,  1,  1,  3,  3,  4,  4,  5,  5,  7 };

    auto single_haystack_column = column_wrapper<single_element_type>(haystack_data, make_validity_initializer(haystack_validity));
    auto single_needle_column   = column_wrapper<single_element_type>(needle_data, make_validity_initializer(needle_validity));
    auto results                = column_wrapper<gdf_size_type      >(needle_data.size(), non_nullable);

//    self_titled_print(single_haystack_column);
//    self_titled_print(single_needle_column);

    gdf_column* haystack_columns[] = { single_haystack_column.get() };
    gdf_column* needle_columns[]   = { single_needle_column.get()   };

    ASSERT_CUDF_SUCCESS(fill(*results.get(), single_element_type{0xDEADBEEF}));

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater,
            nulls_appear_after_values,
            use_haystack_length_for_not_found)
    );

    auto first_greater_than = column_wrapper<gdf_size_type>(first_greater_than_data);

//    print(results, "actual results");
//    print(first_greater_than, "expected results");

    expect_column(results, first_greater_than, print_all_unequal_pairs);

}

TEST(Multisearch, single_column_multiple_needles__find_first_greater_or_equal__nulls_after_other_values)
{
    const auto print_all_unequal_pairs { false };
    using single_element_type = int32_t;
    gdf_column_index_type num_columns     { 1 };

    std::vector<single_element_type> haystack_data     { 10, 20, 30, 40, 50, 10, 60 };
    std::vector<gdf_valid_type     > haystack_validity {  1,  1,  1,  1,  1,  0,  0 };
    std::vector<single_element_type> needle_data       {  8, 10, 11, 30, 32, 40, 47, 50, 90,  8 };
    std::vector<single_element_type> needle_validity   {  1,  1,  1,  1,  1,  1,  1,  1,  1,  0 };
    std::vector<gdf_size_type> first_greater_or_equal_to_data
                                                       {  0,  0,  1,  2,  3,  3,  4,  4,  5,  5 };

    auto single_haystack_column = column_wrapper<single_element_type>(haystack_data, make_validity_initializer(haystack_validity));
    auto single_needle_column   = column_wrapper<single_element_type>(needle_data, make_validity_initializer(needle_validity));
    auto results                = column_wrapper<gdf_size_type      >(needle_data.size(), non_nullable);

//    self_titled_print(single_haystack_column);
//    self_titled_print(single_needle_column);

    gdf_column* haystack_columns[] = { single_haystack_column.get() };
    gdf_column* needle_columns[]   = { single_needle_column.get()   };

    ASSERT_CUDF_SUCCESS(fill(*results.get(), single_element_type{0xDEADBEEF}));

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater_or_equal,
            nulls_appear_after_values,
            use_haystack_length_for_not_found)
    );

    auto first_greater_or_equal_to = column_wrapper<gdf_size_type>(first_greater_or_equal_to_data);

//    print(results, "actual results");
//    print(first_greater_or_equal_to, "expected results");

    expect_column(results, first_greater_or_equal_to, print_all_unequal_pairs);
}

TEST(Multisearch, use_of_nulls_for_not_found)
{
    using single_element_type = int32_t;

    gdf_column_index_type num_columns     { 1 };

    std::vector<single_element_type> haystack_data { 10, 20, 30, 40, 50 };
    std::vector<single_element_type> needle_data              {    0,   50,  100,   50,   25,   50,  100,   50 };
    std::vector<gdf_size_type> first_greater_than_data        {    0, 1234, 5678, 9101,    2, 2345, 6789, 1011 };
    std::vector<gdf_size_type> first_greater_or_equal_to_data {    0,    4, 2222,    4,    2,    4, 3333,    4 };
        // Note the use of dummy data
    std::vector<gdf_valid_type> first_greater_than_validity        { 1, 0, 0, 0, 1, 0, 0, 0 };
    std::vector<gdf_valid_type> first_greater_or_equal_to_validity { 1, 1, 0, 1, 1, 1, 0, 1 };

    auto single_haystack_column = column_wrapper<single_element_type>(haystack_data);
    auto single_needle_column   = column_wrapper<single_element_type>(needle_data);
    auto results                = column_wrapper<gdf_size_type      >(needle_data.size(), nullable);

    gdf_column* haystack_columns[] = { single_haystack_column.get() };
    gdf_column* needle_columns[]   = { single_needle_column.get()   };

//    self_titled_print(single_haystack_column);
//    self_titled_print(single_needle_column);

    ASSERT_CUDF_SUCCESS( fill(*results.get(), gdf_size_type{0xDEADBEEF}) );

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater,
            nulls_appear_before_values,
            use_null_for_not_found);
    );

    auto first_greater_than        = column_wrapper<gdf_size_type>(first_greater_than_data, make_validity_initializer(first_greater_than_validity));

//    print(results, "actual results");
//    print(first_greater_than, "expected results");

    expect_column(results, first_greater_than);

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            &(haystack_columns[0]),
            &(needle_columns[0]),
            num_columns,
            find_first_greater_or_equal,
            nulls_appear_before_values,
            use_null_for_not_found);
    );

    auto first_greater_or_equal_to = column_wrapper<gdf_size_type>(first_greater_or_equal_to_data, make_validity_initializer(first_greater_or_equal_to_validity));

//    print(results, actual results");
//    print(first_greater_or_equal_to, expected results");

    expect_column(results, first_greater_or_equal_to);
}

TEST(Multisearch, multiple_columns_multiple_needles)
{
    using element_types = std::tuple<int32_t, float, int8_t>;
    using type_0 = typename std::tuple_element<0, element_types>::type;
    using type_1 = typename std::tuple_element<1, element_types>::type;
    using type_2 = typename std::tuple_element<2, element_types>::type;

    gdf_column_index_type num_columns { std::tuple_size<element_types>::value };

    std::vector<type_0> haystack_data_0  {  10,  20,  20,  20,  20,  20,  50 };
    std::vector<type_1> haystack_data_1  { 5.0,  .5,  .5,  .7,  .7,  .7,  .7 };
    std::vector<type_2> haystack_data_2  {  90,  77,  78,  61,  62,  63,  41 };

    std::vector<type_0> needle_data_0  { 0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10, 11, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 60 };
    std::vector<type_1> needle_data_1  { 0,  0,  6,  5,  0,  5,  5,  5,  5,  6,  6,  6,  9,  0, .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5 };
    std::vector<type_2> needle_data_2  { 0, 91,  0, 91,  0, 79, 90, 91, 77, 80, 90, 91, 91,  0, 76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20 };


    std::vector<gdf_size_type> first_greater_than_data
                                       { 0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  1,  2,  3,  1,  3,  3,  3,  6,  5,  6,  6,  7,  7 };
    std::vector<gdf_size_type> first_greater_or_equal_to_data
                                       { 0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1,  2,  1,  3,  3,  3,  6,  4,  6,  6,  6,  7 };

    auto haystack_column_wrappers = std::make_tuple(
        column_wrapper<type_0>(haystack_data_0),
        column_wrapper<type_1>(haystack_data_1),
        column_wrapper<type_2>(haystack_data_2)
    );
    auto needle_column_wrappers = std::make_tuple(
        column_wrapper<type_0>(needle_data_0),
        column_wrapper<type_1>(needle_data_1),
        column_wrapper<type_2>(needle_data_2)
    );
    auto results                = column_wrapper<gdf_size_type      >(needle_data_0.size(), non_nullable);

    gdf_column* haystack_columns[] = {
        get<0>(haystack_column_wrappers).get(),
        get<1>(haystack_column_wrappers).get(),
        get<2>(haystack_column_wrappers).get()
    };
    gdf_column* needle_columns[] = {
        get<0>(needle_column_wrappers).get(),
        get<1>(needle_column_wrappers).get(),
        get<2>(needle_column_wrappers).get()
    };

    ASSERT_CUDF_SUCCESS( fill(*results.get(), gdf_size_type{0xDEADBEEF}) );

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            haystack_columns,
            needle_columns,
            num_columns,
            find_first_greater,
            nulls_appear_before_values,
            use_haystack_length_for_not_found)
    );

    auto first_greater_than        = column_wrapper<gdf_size_type>(first_greater_than_data);

//    print(results, "actual results");
//    print(first_greater_than, "expected results");

    expect_column(results, first_greater_than);

    ASSERT_CUDF_SUCCESS(
        gdf_multisearch(
            results.get(),
            haystack_columns,
            needle_columns,
            num_columns,
            find_first_greater_or_equal,
            nulls_appear_before_values,
            use_haystack_length_for_not_found)
    );

    auto first_greater_or_equal_to = column_wrapper<gdf_size_type>(first_greater_or_equal_to_data);

//    print(results, actual results");
//    print(first_greater_or_equal_to, expected results");

    expect_column(results, first_greater_or_equal_to);
}
