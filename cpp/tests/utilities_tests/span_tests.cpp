/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cstddef>
#include <cstring>
#include <string>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/utilities/span.hpp>

using cudf::detail::span;

template <typename T>
void expect_equivolent(span<T> a, span<T> b)
{
  EXPECT_EQ(a.size(), b.size());
  EXPECT_EQ(a.data(), b.data());
}

template <typename Iterator1, typename T>
void expect_match(Iterator1 expected, size_t expected_size, span<T> input)
{
  EXPECT_EQ(expected_size, input.size());
  for (size_t i = 0; i < expected_size; i++) { EXPECT_EQ(*(expected + i), *(input.begin() + i)); }
}

template <typename T>
void expect_match(std::string expected, span<T> input)
{
  return expect_match(expected.begin(), expected.size(), input);
}

template <typename T>
class SpanTest : public cudf::test::BaseFixture {
 public:
  SpanTest() : _hello_world("hello world") {}

  std::vector<char> hello_world()
  {
    return std::vector<char>(_hello_world.begin(), _hello_world.end());
  }

 private:
  std::string const _hello_world;
};

TYPED_TEST_CASE(SpanTest, cudf::test::FloatingPointTypes);

TYPED_TEST(SpanTest, CanCreateFullSubspan)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  expect_equivolent(message_span, message_span.subspan(0, message_span.size()));
}

TYPED_TEST(SpanTest, CanTakeFirst)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  expect_match("hello", message_span.first(5));
}

TYPED_TEST(SpanTest, CanTakeLast)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  expect_match("world", message_span.last(5));
}

TYPED_TEST(SpanTest, CanTakeSubspanFull)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  expect_match("hello world", message_span.subspan(0, 11));
}

TYPED_TEST(SpanTest, CanTakeSubspanPartial)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  expect_match("lo w", message_span.subspan(3, 4));
}

TYPED_TEST(SpanTest, CanGetFront)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  EXPECT_EQ('h', message_span.front());
}

TYPED_TEST(SpanTest, CanGetBack)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  EXPECT_EQ('d', message_span.back());
}

TYPED_TEST(SpanTest, CanGetData)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  EXPECT_EQ(message.data(), message_span.data());
}

TYPED_TEST(SpanTest, CanDetermineEmptiness)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());
  auto const empty_span   = span<char>();

  EXPECT_FALSE(message_span.empty());
  EXPECT_TRUE(empty_span.empty());
}

TYPED_TEST(SpanTest, CanGetSize)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());
  auto const empty_span   = span<char>();

  EXPECT_EQ(11, message_span.size());
  EXPECT_EQ(0, empty_span.size());
}

TYPED_TEST(SpanTest, CanGetSizeBytes)
{
  auto doubles            = std::vector<double>({6, 3, 2});
  auto const doubles_span = span<double>(doubles.data(), doubles.size());
  auto const empty_span   = span<double>();

  EXPECT_EQ(24, doubles_span.size_bytes());
  EXPECT_EQ(0, empty_span.size_bytes());
}

TYPED_TEST(SpanTest, CanCopySpan)
{
  auto message = this->hello_world();
  span<char> message_span_copy;

  {
    auto const message_span = span<char>(message.data(), message.size());

    message_span_copy = message_span;
  }

  EXPECT_EQ(message.data(), message_span_copy.data());
  EXPECT_EQ(message.size(), message_span_copy.size());
}

TYPED_TEST(SpanTest, CanSubscriptRead)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  EXPECT_EQ('o', message_span[4]);
}

TYPED_TEST(SpanTest, CanSubscriptWrite)
{
  auto message            = this->hello_world();
  auto const message_span = span<char>(message.data(), message.size());

  message_span[4] = 'x';

  EXPECT_EQ('x', message_span[4]);
}
