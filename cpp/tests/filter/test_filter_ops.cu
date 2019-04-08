/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
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

#include "test_filter_ops.cuh"

template <typename DataColumnElement>
struct ApplyBooleanMaskTest : public GdfTest {

    using element_type = DataColumnElement;

    ApplyBooleanMaskTest() = default;
    ~ApplyBooleanMaskTest() = default;

};

typedef ::testing::Types<
    int8_t,
    int32_t,
    int64_t,
    float,
    double
  > Implementations;

TYPED_TEST_CASE(ApplyBooleanMaskTest, Implementations);

//Todo: usage_example
//TYPED_TEST(ApplyBooleanMaskTest, usage_example) {


TYPED_TEST(ApplyBooleanMaskTest, all_multiple_32) {
    using element_type = typename TestFixture::element_type;

    constexpr const auto size =  column_sizes::short_round;


    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},    fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(size, &constant<gdf_bool, gdf_true>,            fully_valid);
    auto output  = column_wrapper<element_type>(size, &zero<element_type>,                      fully_valid);

    auto expected_output = compute_expected_output(data, stencil);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, all_non_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;
    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{}, fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(size, constant<gdf_bool, gdf_true>,          fully_valid);
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                    fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, half_all_third_multiple_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{}, first_half{size});
    auto stencil = column_wrapper<gdf_bool    >(size, constant<gdf_bool, gdf_true>,          first_of_every<3>{});
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                    fully_valid);


    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, half_all_third_non_multiple_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{}, first_half{size});
    auto stencil = column_wrapper<gdf_bool    >(size, constant<gdf_bool, gdf_true>,          first_of_every<3>{});
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                    fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);

    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, all_half_all_non_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{}, fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(size, first_half{size},                    fully_valid);
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                    fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, all_random_all_non_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(size, uniformly_distributed<gdf_bool>{0,1},   fully_valid);
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);


    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, all_third_random_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(size, first_of_every<3>{},                    uniformly_distributed<char>{0,1});
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, random_random_random_non_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  uniformly_distributed<char>{0,1});
    auto stencil = column_wrapper<gdf_bool    >(size, uniformly_distributed<gdf_bool>{0, 1},  uniformly_distributed<char>{0,1});
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);

    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, non_nullable_all_all_non_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;

//    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  non_nullable);

    auto data { make_non_nullable_column_wrapper(size, uniformly_distributed<element_type>{}) };
    auto stencil = column_wrapper<gdf_bool    >(size, constant<gdf_bool, gdf_true>,           fully_valid);
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);

    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, all_none_all_non_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(size, constant<gdf_bool, gdf_false>,          fully_valid);
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);

    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, all_all_none_non_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  fully_valid);
//    auto stencil = column_wrapper<gdf_bool    >(size, constant<gdf_bool, gdf_true>,           non_nullable);
    auto stencil = make_non_nullable_column_wrapper(size, constant<gdf_bool, gdf_true>);
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, none_none_none_non_multiple_of_32) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::short_non_round;

//    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  non_nullable);
//    auto stencil = column_wrapper<gdf_bool    >(size, constant<gdf_bool, gdf_false>{},        non_nullable);
    auto data    = make_non_nullable_column_wrapper(size, uniformly_distributed<element_type>{});
    auto stencil = make_non_nullable_column_wrapper(size, constant<gdf_bool, gdf_false>);

    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);
}

TYPED_TEST(ApplyBooleanMaskTest, all_random_random_big_input) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size =  column_sizes::long_non_round;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(size, uniformly_distributed<gdf_bool>{},      uniformly_distributed<char>{0,1});
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);

}

TYPED_TEST(ApplyBooleanMaskTest, all_random_random_empty_input) {
    using element_type = typename TestFixture::element_type;
    constexpr const auto size = 0;

    auto data    = column_wrapper<element_type>(size, uniformly_distributed<element_type>{},  fully_valid);
    auto stencil = column_wrapper<gdf_bool    >(size, uniformly_distributed<gdf_bool>{},      uniformly_distributed<char>{0,1});
    auto output  = column_wrapper<element_type>(size, zero<element_type>,                     fully_valid);

    ASSERT_CUDF_SUCCESS( gdf_apply_boolean_mask(data.get(), stencil.get(), output.get()) );
    auto expected_output = compute_expected_output(data, stencil);
    expect_columns_are_equal(output, expected_output);
}
