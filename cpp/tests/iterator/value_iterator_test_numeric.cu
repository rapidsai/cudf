/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/value_iterator_test.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/type_lists.hpp>

using TestingTypes = cudf::test::NumericTypes;

template <typename T>
struct NumericValueIteratorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(NumericValueIteratorTest, TestingTypes);
TYPED_TEST(NumericValueIteratorTest, non_null_iterator) { non_null_iterator(*this); }
TYPED_TEST(NumericValueIteratorTest, null_iterator) { null_iterator(*this); }
