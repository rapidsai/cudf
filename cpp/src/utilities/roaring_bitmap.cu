/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/roaring_bitmap.cuh>
#include <cuda/iterator>

namespace cudf {

namespace {

using roaring_bitmap_32_type =
  cuco::experimental::roaring_bitmap<cuda::std::uint32_t, rmm::mr::polymorphic_allocator<char>>;
using roaring_bitmap_64_type =
  cuco::experimental::roaring_bitmap<cuda::std::uint64_t, rmm::mr::polymorphic_allocator<char>>;

}  // namespace

struct roaring_bitmap::roaring_bitmap_impl {
  cudf::host_span<cuda::std::byte const> _serialized_bitmap_data;
  std::unique_ptr<roaring_bitmap_32_type> _bitmap32;
  std::unique_ptr<roaring_bitmap_64_type> _bitmap64;

  roaring_bitmap_impl(cudf::host_span<cuda::std::byte const> serialized_bitmap_data)
    : _serialized_bitmap_data{serialized_bitmap_data}
  {
  }

  roaring_bitmap_impl(roaring_bitmap_impl&&)            = default;
  roaring_bitmap_impl& operator=(roaring_bitmap_impl&&) = default;

  template <roaring_bitmap_type Type>
  void materialize(rmm::cuda_stream_view stream)
    requires(Type == roaring_bitmap_type::BITS_32 or Type == roaring_bitmap_type::BITS_64)
  {
    auto const bytes = _serialized_bitmap_data.data();

    if constexpr (Type == roaring_bitmap_type::BITS_32) {
      if (_bitmap32) { return; }
      _bitmap32 = std::make_unique<roaring_bitmap_32_type>(
        bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
    } else {
      if (_bitmap64) { return; }
      _bitmap64 = std::make_unique<roaring_bitmap_64_type>(
        bytes, rmm::mr::polymorphic_allocator<char>{}, stream);
    }

    _serialized_bitmap_data = {};
  }

  template <roaring_bitmap_type Type, typename InputIt, typename OutputIt>
  void contains_async(InputIt first, InputIt last, OutputIt output, rmm::cuda_stream_view stream)
    requires(Type == roaring_bitmap_type::BITS_32 or Type == roaring_bitmap_type::BITS_64)
  {
    // Return early if the input range is empty
    if (first == last) { return; }

    materialize<Type>(stream);

    if constexpr (Type == roaring_bitmap_type::BITS_32) {
      CUDF_EXPECTS(_bitmap32,
                   "Roaring bitmap has not been materialized. Call materialize() first.");
      _bitmap32->contains_async(first, last, output, stream);
    } else {
      CUDF_EXPECTS(_bitmap64,
                   "Roaring bitmap has not been materialized. Call materialize() first.");
      _bitmap64->contains_async(first, last, output, stream);
    }
  }
};

roaring_bitmap::roaring_bitmap(roaring_bitmap_type type,
                               cudf::host_span<cuda::std::byte const> serialized_bitmap_data)
  : _type{type}
{
  CUDF_EXPECTS(not serialized_bitmap_data.empty(),
               "Encountered empty serialized roaring bitmap data",
               std::invalid_argument);
  _impl = std::make_unique<roaring_bitmap_impl>(serialized_bitmap_data);
}

roaring_bitmap::~roaring_bitmap() = default;

roaring_bitmap::roaring_bitmap(roaring_bitmap&&) noexcept = default;

roaring_bitmap& roaring_bitmap::operator=(roaring_bitmap&&) noexcept = default;

void roaring_bitmap::materialize(rmm::cuda_stream_view stream) const
{
  if (_type == roaring_bitmap_type::BITS_32) {
    _impl->materialize<roaring_bitmap_type::BITS_32>(stream);
  } else {
    _impl->materialize<roaring_bitmap_type::BITS_64>(stream);
  }
}

std::unique_ptr<cudf::column> roaring_bitmap::contains_async(
  cudf::column_view const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto result = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, keys.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  contains_async(keys, result->mutable_view(), stream);
  return result;
}

void roaring_bitmap::contains_async(cudf::column_view const& keys,
                                    cudf::mutable_column_view const& output,
                                    rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(output.type().id() == cudf::type_id::BOOL8, "Output column must be BOOL8");
  CUDF_EXPECTS(output.size() >= keys.size(), "Output column size must be >= keys column size");

  if (_type == roaring_bitmap_type::BITS_32) {
    CUDF_EXPECTS(keys.type().id() == cudf::type_id::UINT32,
                 "Key column must be UINT32 for a 32-bit roaring bitmap",
                 std::invalid_argument);
    _impl->contains_async<roaring_bitmap_type::BITS_32>(keys.begin<cuda::std::uint32_t>(),
                                                        keys.end<cuda::std::uint32_t>(),
                                                        output.begin<bool>(),
                                                        stream);
  } else {
    CUDF_EXPECTS(keys.type().id() == cudf::type_id::UINT64,
                 "Key column must be UINT64 for a 64-bit roaring bitmap",
                 std::invalid_argument);
    _impl->contains_async<roaring_bitmap_type::BITS_64>(keys.begin<cuda::std::uint64_t>(),
                                                        keys.end<cuda::std::uint64_t>(),
                                                        output.begin<bool>(),
                                                        stream);
  }
}

std::unique_ptr<cudf::column> roaring_bitmap::not_contains_async(
  cudf::column_view const& keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto result = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, keys.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  not_contains_async(keys, result->mutable_view(), stream);
  return result;
}

void roaring_bitmap::not_contains_async(cudf::column_view const& keys,
                                        cudf::mutable_column_view const& output,
                                        rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(output.type().id() == cudf::type_id::BOOL8, "Output column must be BOOL8");
  CUDF_EXPECTS(output.size() >= keys.size(), "Output column size must be >= keys column size");

  auto output_iter =
    cuda::make_transform_output_iterator(output.begin<bool>(), cuda::std::logical_not<bool>{});

  if (_type == roaring_bitmap_type::BITS_32) {
    CUDF_EXPECTS(keys.type().id() == cudf::type_id::UINT32,
                 "Key column must be UINT32 for a 32-bit roaring bitmap",
                 std::invalid_argument);
    _impl->contains_async<roaring_bitmap_type::BITS_32>(
      keys.begin<cuda::std::uint32_t>(), keys.end<cuda::std::uint32_t>(), output_iter, stream);
  } else {
    CUDF_EXPECTS(keys.type().id() == cudf::type_id::UINT64,
                 "Key column must be UINT64 for a 64-bit roaring bitmap",
                 std::invalid_argument);
    _impl->contains_async<roaring_bitmap_type::BITS_64>(
      keys.begin<cuda::std::uint64_t>(), keys.end<cuda::std::uint64_t>(), output_iter, stream);
  }
}

roaring_bitmap_type roaring_bitmap::type() const { return _type; }

}  // namespace cudf
