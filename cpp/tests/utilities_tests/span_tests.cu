/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <cstring>
#include <string>

using cudf::device_span;
using cudf::host_span;
using cudf::detail::device_2dspan;
using cudf::detail::host_2dspan;
using cudf::detail::hostdevice_2dvector;

template <typename T>
void expect_equivolent(host_span<T> a, host_span<T> b)
{
  EXPECT_EQ(a.size(), b.size());
  EXPECT_EQ(a.data(), b.data());
}

template <typename Iterator1, typename T>
void expect_match(Iterator1 expected, size_t expected_size, host_span<T> input)
{
  EXPECT_EQ(expected_size, input.size());
  for (size_t i = 0; i < expected_size; i++) {
    EXPECT_EQ(*(expected + i), *(input.begin() + i));
  }
}

template <typename T>
void expect_match(std::string expected, host_span<T> input)
{
  return expect_match(expected.begin(), expected.size(), input);
}

std::string const hello_wold_message = "hello world";
std::vector<char> create_hello_world_message()
{
  return std::vector<char>(hello_wold_message.begin(), hello_wold_message.end());
}

class SpanTest : public cudf::test::BaseFixture {
};

TEST(SpanTest, CanCreateFullSubspan)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_equivolent(message_span, message_span.subspan(0, message_span.size()));
}

TEST(SpanTest, CanTakeFirst)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_match("hello", message_span.first(5));
}

TEST(SpanTest, CanTakeLast)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_match("world", message_span.last(5));
}

TEST(SpanTest, CanTakeSubspanFull)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_match("hello world", message_span.subspan(0, 11));
}

TEST(SpanTest, CanTakeSubspanPartial)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_match("lo w", message_span.subspan(3, 4));
}

TEST(SpanTest, CanGetFront)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  EXPECT_EQ('h', message_span.front());
}

TEST(SpanTest, CanGetBack)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  EXPECT_EQ('d', message_span.back());
}

TEST(SpanTest, CanGetData)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  EXPECT_EQ(message.data(), message_span.data());
}

TEST(SpanTest, CanDetermineEmptiness)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());
  auto const empty_span   = host_span<char>();

  EXPECT_FALSE(message_span.empty());
  EXPECT_TRUE(empty_span.empty());
}

TEST(SpanTest, CanGetSize)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());
  auto const empty_span   = host_span<char>();

  EXPECT_EQ(static_cast<size_t>(11), message_span.size());
  EXPECT_EQ(static_cast<size_t>(0), empty_span.size());
}

TEST(SpanTest, CanGetSizeBytes)
{
  auto doubles            = std::vector<double>({6, 3, 2});
  auto const doubles_span = host_span<double>(doubles.data(), doubles.size());
  auto const empty_span   = host_span<double>();

  EXPECT_EQ(static_cast<size_t>(24), doubles_span.size_bytes());
  EXPECT_EQ(static_cast<size_t>(0), empty_span.size_bytes());
}

TEST(SpanTest, CanCopySpan)
{
  auto message = create_hello_world_message();
  host_span<char> message_span_copy;

  {
    auto const message_span = host_span<char>(message.data(), message.size());

    message_span_copy = message_span;
  }

  EXPECT_EQ(message.data(), message_span_copy.data());
  EXPECT_EQ(message.size(), message_span_copy.size());
}

TEST(SpanTest, CanSubscriptRead)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  EXPECT_EQ('o', message_span[4]);
}

TEST(SpanTest, CanSubscriptWrite)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  message_span[4] = 'x';

  EXPECT_EQ('x', message_span[4]);
}

TEST(SpanTest, CanConstructFromHostContainers)
{
  auto std_vector = std::vector<int>(1);
  auto h_vector   = thrust::host_vector<int>(1);

  (void)host_span<int>(std_vector);
  (void)host_span<int>(h_vector);

  auto const std_vector_c = std_vector;
  auto const h_vector_c   = h_vector;

  (void)host_span<int const>(std_vector_c);
  (void)host_span<int const>(h_vector_c);
}

TEST(SpanTest, CanConstructFromDeviceContainers)
{
  auto d_thrust_vector = thrust::device_vector<int>(1);
  auto d_vector        = rmm::device_vector<int>(1);
  auto d_uvector       = rmm::device_uvector<int>(1, cudf::default_stream_value);

  (void)device_span<int>(d_thrust_vector);
  (void)device_span<int>(d_vector);
  (void)device_span<int>(d_uvector);

  auto const& d_thrust_vector_c = d_thrust_vector;
  auto const& d_vector_c        = d_vector;
  auto const& d_uvector_c       = d_uvector;

  (void)device_span<int const>(d_thrust_vector_c);
  (void)device_span<int const>(d_vector_c);
  (void)device_span<int const>(d_uvector_c);
}

__global__ void simple_device_kernel(device_span<bool> result) { result[0] = true; }

TEST(SpanTest, CanUseDeviceSpan)
{
  rmm::device_vector<bool> d_message = std::vector<bool>({false});

  auto d_span = device_span<bool>(d_message.data().get(), d_message.size());

  simple_device_kernel<<<1, 1, 0, cudf::default_stream_value.value()>>>(d_span);

  cudaDeviceSynchronize();

  thrust::host_vector<bool> h_message = d_message;

  ASSERT_TRUE(h_message[0]);
}

class MdSpanTest : public cudf::test::BaseFixture {
};

TEST(MdSpanTest, CanDetermineEmptiness)
{
  auto const vector            = hostdevice_2dvector<int>(1, 2, cudf::default_stream_value);
  auto const no_rows_vector    = hostdevice_2dvector<int>(0, 2, cudf::default_stream_value);
  auto const no_columns_vector = hostdevice_2dvector<int>(1, 0, cudf::default_stream_value);

  EXPECT_FALSE(host_2dspan<int const>{vector}.is_empty());
  EXPECT_FALSE(device_2dspan<int const>{vector}.is_empty());
  EXPECT_TRUE(host_2dspan<int const>{no_rows_vector}.is_empty());
  EXPECT_TRUE(device_2dspan<int const>{no_rows_vector}.is_empty());
  EXPECT_TRUE(host_2dspan<int const>{no_columns_vector}.is_empty());
  EXPECT_TRUE(device_2dspan<int const>{no_columns_vector}.is_empty());
}

__global__ void readwrite_kernel(device_2dspan<int> result)
{
  if (result[5][6] == 5) {
    result[5][6] *= 6;
  } else {
    result[5][6] = 5;
  }
}

TEST(MdSpanTest, DeviceReadWrite)
{
  auto vector = hostdevice_2dvector<int>(11, 23, cudf::default_stream_value);

  readwrite_kernel<<<1, 1, 0, cudf::default_stream_value.value()>>>(vector);
  readwrite_kernel<<<1, 1, 0, cudf::default_stream_value.value()>>>(vector);
  vector.device_to_host(cudf::default_stream_value, true);
  EXPECT_EQ(vector[5][6], 30);
}

TEST(MdSpanTest, HostReadWrite)
{
  auto vector = hostdevice_2dvector<int>(11, 23, cudf::default_stream_value);
  auto span   = host_2dspan<int>{vector};
  span[5][6]  = 5;
  if (span[5][6] == 5) { span[5][6] *= 6; }

  EXPECT_EQ(vector[5][6], 30);
}

TEST(MdSpanTest, CanGetSize)
{
  auto const vector = hostdevice_2dvector<int>(1, 2, cudf::default_stream_value);

  EXPECT_EQ(host_2dspan<int const>{vector}.size(), vector.size());
  EXPECT_EQ(device_2dspan<int const>{vector}.size(), vector.size());
}

TEST(MdSpanTest, CanGetCount)
{
  auto const vector = hostdevice_2dvector<int>(11, 23, cudf::default_stream_value);

  EXPECT_EQ(host_2dspan<int const>{vector}.count(), 11ul * 23);
  EXPECT_EQ(device_2dspan<int const>{vector}.count(), 11ul * 23);
}

CUDF_TEST_PROGRAM_MAIN()
