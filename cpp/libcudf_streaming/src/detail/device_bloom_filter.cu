/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// cuco headers have sign-conversion issues; suppress for the host compiler
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <cuco/bloom_filter_policies.cuh>
#include <cuco/bloom_filter_ref.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cudf_streaming/detail/device_bloom_filter.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_transform.cuh>
#include <cuda/std/tuple>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>

namespace cudf_streaming::detail {

namespace {
using KeyType = std::uint64_t;

using BloomFilterRefType =
  cuco::bloom_filter_ref<KeyType,
                         cuco::extent<std::size_t>,
                         cuco::thread_scope_device,
                         cuco::parametric_filter_policy<cuco::identity_hash<KeyType>,
                                                        std::uint32_t,
                                                        8,
                                                        8,
                                                        8,
                                                        1,
                                                        1,
                                                        8,
                                                        false,
                                                        false>>;
using StorageType = BloomFilterRefType::filter_block_type;

}  // namespace

device_bloom_filter::device_bloom_filter(std::size_t num_blocks,
                                         std::uint64_t seed,
                                         void* storage,
                                         rmm::cuda_stream_view stream)
  : num_blocks_{num_blocks}, seed_{seed}, storage_{storage}, stream_{stream}
{
  // TODO: use an aligned allocator adaptor to ensure this holds.
  // Today all RMM device allocators guarantee at least 256 byte alignment, but that is
  // an implementation detail.
  RAPIDSMPF_EXPECTS(
    reinterpret_cast<std::uintptr_t>(storage_) % std::alignment_of_v<StorageType> == 0,
    "Allocation for bloom filter is not aligned.");
}

device_bloom_filter const device_bloom_filter::view(std::size_t num_blocks,
                                                    std::uint64_t seed,
                                                    void const* storage,
                                                    rmm::cuda_stream_view stream)
{
  // const-cast is safe because the returned object is also const and therefore can't
  // call methods that throw away constness.
  return device_bloom_filter(num_blocks, seed, const_cast<void*>(storage), stream);
}

std::unique_ptr<rmm::device_buffer> device_bloom_filter::storage(std::size_t num_blocks,
                                                                 rmm::cuda_stream_view stream,
                                                                 rmm::device_async_resource_ref mr)
{
  return std::make_unique<rmm::device_buffer>(num_blocks * sizeof(StorageType), stream, mr);
}

void device_bloom_filter::add(cudf::table_view const& values_to_hash,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  auto filter_ref = BloomFilterRefType{
    static_cast<StorageType*>(storage_), num_blocks_, cuco::thread_scope_device, {}};
  auto hashes = cudf::hashing::xxhash_64(
    values_to_hash, seed_, stream, cudf::get_current_device_resource_ref());
  auto hash_view = hashes->view();
  RAPIDSMPF_EXPECTS(hash_view.type().id() == cudf::type_to_id<KeyType>(),
                    "Hash values do not have correct type");
  filter_ref.add_async(hash_view.begin<KeyType>(), hash_view.end<KeyType>(), stream);
}

void device_bloom_filter::merge(device_bloom_filter const& other, rmm::cuda_stream_view stream)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  RAPIDSMPF_EXPECTS(num_blocks_ == other.num_blocks_, "Mismatching number of blocks in filters");
  auto ref_this = BloomFilterRefType{
    static_cast<StorageType*>(storage_), num_blocks_, cuco::thread_scope_device, {}};
  auto ref_other = BloomFilterRefType{
    static_cast<StorageType*>(other.storage_), num_blocks_, cuco::thread_scope_device, {}};
  ref_this.merge_async(ref_other, stream);
}

rmm::device_uvector<bool> device_bloom_filter::contains(cudf::table_view const& values,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr) const
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  auto filter_ref = BloomFilterRefType{
    static_cast<StorageType*>(storage_), num_blocks_, cuco::thread_scope_device, {}};
  auto hashes =
    cudf::hashing::xxhash_64(values, seed_, stream, cudf::get_current_device_resource_ref());
  auto view = hashes->view();
  rmm::device_uvector<bool> result{static_cast<std::size_t>(view.size()), stream, mr};
  filter_ref.contains_async(view.begin<KeyType>(), view.end<KeyType>(), result.begin(), stream);
  return result;
}

std::size_t device_bloom_filter::fitting_num_blocks(std::size_t l2size) noexcept
{
  return (l2size * 2) / (3 * sizeof(StorageType));
}

rmm::cuda_stream_view device_bloom_filter::stream() const noexcept { return stream_; }

void* device_bloom_filter::data() noexcept { return storage_; }

void const* device_bloom_filter::data() const noexcept { return storage_; }

std::size_t device_bloom_filter::size() const noexcept { return num_blocks_ * sizeof(StorageType); }

}  // namespace cudf_streaming::detail
