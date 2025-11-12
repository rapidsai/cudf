/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/value_iterator_test.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/type_lists.hpp>

using TestingTypes = cudf::test::ChronoTypes;

template <typename T>
struct ChronoValueIteratorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(ChronoValueIteratorTest, TestingTypes);
TYPED_TEST(ChronoValueIteratorTest, non_null_iterator) { non_null_iterator(*this); }
TYPED_TEST(ChronoValueIteratorTest, null_iterator) { null_iterator(*this); }
