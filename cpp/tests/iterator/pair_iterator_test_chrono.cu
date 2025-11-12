/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/pair_iterator_test.cuh>

using TestingTypes = cudf::test::ChronoTypes;

template <typename T>
struct ChronoPairIteratorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(ChronoPairIteratorTest, TestingTypes);
TYPED_TEST(ChronoPairIteratorTest, nonull_pair_iterator) { nonull_pair_iterator(*this); }
TYPED_TEST(ChronoPairIteratorTest, null_pair_iterator) { null_pair_iterator(*this); }
