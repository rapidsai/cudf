
#pragma once

#include <cub/block/block_scan.cuh>

namespace cudf {
namespace io {
namespace text {
namespace detail {

enum class scan_tile_status : uint8_t {
  oob,
  invalid,
  partial,
  inclusive,
};

template <typename T>
struct scan_tile_state_view {
  uint64_t num_tiles;
  scan_tile_status* tile_status;
  T* tile_partial;
  T* tile_inclusive;

  __device__ inline void initialize_status(cudf::size_type base_tile_idx,
                                           cudf::size_type count,
                                           scan_tile_status status)
  {
    auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx < count) {  //
      // this is UB if tile_status gets assigned from multiple threads.
      tile_status[(base_tile_idx + thread_idx) % num_tiles] = status;
    }
  }

  __device__ inline void set_partial_prefix(cudf::size_type tile_idx, T value)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    cub::ThreadStore<cub::STORE_CG>(tile_partial + offset, value);
    __threadfence();
    cub::ThreadStore<cub::STORE_CG>(tile_status + offset, scan_tile_status::partial);
  }

  __device__ inline void set_inclusive_prefix(cudf::size_type tile_idx, T value)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    cub::ThreadStore<cub::STORE_CG>(tile_inclusive + offset, value);
    __threadfence();
    cub::ThreadStore<cub::STORE_CG>(tile_status + offset, scan_tile_status::inclusive);
  }

  __device__ inline T get_prefix(cudf::size_type tile_idx, scan_tile_status& status)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;

    while ((status = cub::ThreadLoad<cub::LOAD_CG>(tile_status + offset)) ==
           scan_tile_status::invalid) {
      __threadfence();
    }

    if (status == scan_tile_status::partial) {
      return cub::ThreadLoad<cub::LOAD_CG>(tile_partial + offset);
    } else {
      return cub::ThreadLoad<cub::LOAD_CG>(tile_inclusive + offset);
    }
  }

  __device__ inline T get_inclusive_prefix(cudf::size_type tile_idx)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    while (cub::ThreadLoad<cub::LOAD_CG>(tile_status + offset) != scan_tile_status::inclusive) {
      __threadfence();
    }
    return cub::ThreadLoad<cub::LOAD_CG>(tile_inclusive + offset);
  }
};

template <typename T>
struct scan_tile_state {
  rmm::device_uvector<scan_tile_status> tile_status;
  rmm::device_uvector<T> tile_state_partial;
  rmm::device_uvector<T> tile_state_inclusive;

  scan_tile_state(cudf::size_type num_tiles,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : tile_status(rmm::device_uvector<scan_tile_status>(num_tiles, stream, mr)),
      tile_state_partial(rmm::device_uvector<T>(num_tiles, stream, mr)),
      tile_state_inclusive(rmm::device_uvector<T>(num_tiles, stream, mr))
  {
  }

  operator scan_tile_state_view<T>()
  {
    return scan_tile_state_view<T>{tile_status.size(),
                                   tile_status.data(),
                                   tile_state_partial.data(),
                                   tile_state_inclusive.data()};
  }

  inline void set_seed_async(T const seed, rmm::cuda_stream_view stream)
  {
    auto x = tile_status.size();
    auto y = scan_tile_status::inclusive;
    tile_state_inclusive.set_element_async(x - 1, seed, stream);
    tile_status.set_element_async(x - 1, y, stream);
  }

  // T back_element(rmm::cuda_stream_view stream) const { return tile_state.back_element(stream); }

  inline T get_inclusive_prefix(cudf::size_type tile_idx, rmm::cuda_stream_view stream) const
  {
    auto const offset = (tile_idx + tile_status.size()) % tile_status.size();
    return tile_state_inclusive.element(offset, stream);
  }
};

template <typename T>
struct scan_tile_state_callback {
  using WarpReduce = cub::WarpReduce<T>;

  struct _TempStorage {
    typename WarpReduce::TempStorage reduce;
    T exclusive_prefix;
  };

  using TempStorage = cub::Uninitialized<_TempStorage>;

  __device__ inline scan_tile_state_callback(TempStorage& temp_storage,
                                             scan_tile_state_view<T>& tile_state,
                                             cudf::size_type tile_idx)
    : _temp_storage(temp_storage.Alias()), _tile_state(tile_state), _tile_idx(tile_idx)
  {
  }

  __device__ inline T operator()(T const& block_aggregate)
  {
    if (threadIdx.x == 0) {
      _tile_state.set_partial_prefix(_tile_idx, block_aggregate);  //
    }

    auto predecessor_idx    = _tile_idx - 1 - threadIdx.x;
    auto predecessor_status = scan_tile_status::invalid;

    // scan partials to form prefix

    if (threadIdx.x == 0) {
      auto window_partial = _tile_state.get_prefix(predecessor_idx, predecessor_status);
      while (predecessor_status != scan_tile_status::inclusive) {
        predecessor_idx--;
        auto predecessor_prefix = _tile_state.get_prefix(predecessor_idx, predecessor_status);
        window_partial          = predecessor_prefix + window_partial;
      }
      _temp_storage.exclusive_prefix = window_partial;
    }

    if (threadIdx.x == 0) {
      _tile_state.set_inclusive_prefix(_tile_idx, _temp_storage.exclusive_prefix + block_aggregate);
    }

    __syncthreads();  // TODO: remove if unnecessary.

    return _temp_storage.exclusive_prefix;
  }

  _TempStorage& _temp_storage;
  scan_tile_state_view<T>& _tile_state;
  cudf::size_type _tile_idx;
};

}  // namespace detail
}  // namespace text
}  // namespace io
}  // namespace cudf
