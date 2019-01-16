/*
 * CachedPinnedAllocator.h
 *
 *  Created on: Mar 15, 2018
 *      Author: felipe
 */

#ifndef SRC_GPUABSTRACTIONS_CACHEDPINNEDALLOCATOR_H_
#define SRC_GPUABSTRACTIONS_CACHEDPINNEDALLOCATOR_H_

/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple devices.
 ******************************************************************************/

#include <map>
#include <set>

#include <math.h>
#include <mutex>

#include "driver_types.h"

/******************************************************************************
 * CachingPinnedAllocator (host use)
 ******************************************************************************/

/**
 * \brief A simple caching allocator for device memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe and stream-safe and is capable of managing cached
 * device allocations on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations from the allocator are associated with an \p active_stream.  Once freed,
 *   the allocation becomes available immediately for reuse within the \p active_stream
 *   with which it was associated with during allocation, and it becomes available for
 *   reuse within other streams when all prior work submitted to \p active_stream has completed.
 * - Allocations are categorized and cached by bin size.  A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused device allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations on a given device will exceed
 *   \p max_cached_bytes, allocations for that device are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingPinnedAllocator is configured with:
 * - \p bin_growth          = 8
 * - \p min_bin             = 3
 * - \p max_bin             = 7
 * - \p max_cached_bytes    = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per device
 *
 */
struct CachingPinnedAllocator {

    //---------------------------------------------------------------------
    // Constants
    //---------------------------------------------------------------------

    /// Out-of-bounds bin
    static const unsigned int INVALID_BIN = (unsigned int) -1;

    /// Invalid size
    static const size_t INVALID_SIZE = (size_t) -1;

    //---------------------------------------------------------------------
    // Type definitions and helper types
    //---------------------------------------------------------------------

    /**
	 * Descriptor for device memory allocations
	 */
    struct BlockDescriptor {
        void *       d_ptr;  // Device pointer
        size_t       bytes;  // Size of allocation in bytes
        unsigned int bin;    // Bin enumeration
        //    int             device;             // device ordinal
        //     cudaStream_t    associated_stream;  // Associated associated_stream
        //     cudaEvent_t     ready_event;        // Signal when associated stream has run to the point at which this block was freed

        // Constructor (suitable for searching maps for a specific block, given its pointer and device)
        BlockDescriptor(void *d_ptr)
          : d_ptr(d_ptr), bytes(0), bin(INVALID_BIN)

        {}

        // Constructor (suitable for searching maps for a range of suitable blocks, given a device)
        BlockDescriptor() : d_ptr(NULL), bytes(0), bin(INVALID_BIN) {}

        // Comparison functor for comparing device pointers
        static bool PtrCompare(const BlockDescriptor &a,
                               const BlockDescriptor &b) {
            return (a.d_ptr < b.d_ptr);
        }

        // Comparison functor for comparing allocation sizes
        static bool SizeCompare(const BlockDescriptor &a,
                                const BlockDescriptor &b) {
            return (a.bytes < b.bytes);
        }
    };

    /// BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

    class TotalBytes {
    public:
        size_t free;
        size_t live;
        TotalBytes() { free = live = 0; }
    };

    /// Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

    /// Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

    /// Map type of device ordinals to the number of cached bytes cached by each device

    //---------------------------------------------------------------------
    // Utility functions
    //---------------------------------------------------------------------

    /**
	 * Integer pow function for unsigned base and exponent
	 */
    static unsigned int IntPow(unsigned int base, unsigned int exp) {
        unsigned int retval = 1;
        while (exp > 0) {
            if (exp & 1) {
                retval =
                  retval * base;  // multiply the result by the current base
            }
            base = base * base;  // square the base
            exp  = exp >> 1;     // divide the exponent in half
        }
        return retval;
    }

    /**
	 * Round up to the nearest power-of
	 */
    void NearestPowerOf(unsigned int &power,
                        size_t &      rounded_bytes,
                        unsigned int  base,
                        size_t        value) {
        power         = 0;
        rounded_bytes = 1;

        if (value * base < value) {
            // Overflow
            power         = sizeof(size_t) * 8;
            rounded_bytes = size_t(0) - 1;
            return;
        }

        while (rounded_bytes < value) {
            rounded_bytes *= base;
            power++;
        }
    }

    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    std::mutex mutex;  /// Mutex for thread-safety

    unsigned int bin_growth;  /// Geometric growth factor for bin-sizes
    unsigned int min_bin;     /// Minimum bin enumeration
    unsigned int max_bin;     /// Maximum bin enumeration

    size_t min_bin_bytes;     /// Minimum bin size
    size_t max_bin_bytes;     /// Maximum bin size
    size_t max_cached_bytes;  /// Maximum aggregate cached bytes per device

    const bool
         skip_cleanup;  /// Whether or not to skip a call to FreeAllCached() when destructor is called.  (The CUDA runtime may have already shut down for statically declared allocators)
    bool debug;  /// Whether or not to print (de)allocation events to stdout

    TotalBytes
      cached_bytes;  /// Map of device ordinal to aggregate cached bytes on that device
    CachedBlocks
               cached_blocks;  /// Set of cached device allocations available for reuse
    BusyBlocks live_blocks;  /// Set of live device allocations currently in use

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
	 * \brief Constructor.
	 */
    CachingPinnedAllocator(
      unsigned int bin_growth,   ///< Geometric growth factor for bin-sizes
      unsigned int min_bin = 1,  ///< Minimum bin (default is bin_growth ^ 1)
      unsigned int max_bin =
        INVALID_BIN,  ///< Maximum bin (default is no max bin)
      size_t max_cached_bytes =
        INVALID_SIZE,  ///< Maximum aggregate cached bytes per device (default is no limit)
      bool skip_cleanup =
        false,  ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called (default is to deallocate)
      bool debug =
        false)  ///< Whether or not to print (de)allocation events to stdout (default is no stderr output)
      : bin_growth(bin_growth), min_bin(min_bin), max_bin(max_bin),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes(max_cached_bytes), skip_cleanup(skip_cleanup),
        debug(debug), cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare) {}

    /**
	 * \brief Default constructor.
	 *
	 * Configured with:
	 * \par
	 * - \p bin_growth          = 8
	 * - \p min_bin             = 3
	 * - \p max_bin             = 7
	 * - \p max_cached_bytes    = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
	 *
	 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
	 * sets a maximum of 6,291,455 cached bytes per device
	 */
    CachingPinnedAllocator(bool skip_cleanup = false, bool debug = false)
      : bin_growth(2), min_bin(20), max_bin(25),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes((max_bin_bytes * 3) - 1), skip_cleanup(skip_cleanup),
        debug(debug), cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare) {}

    /**
	 * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
	 *
	 * Changing the ceiling of cached bytes does not cause any allocations (in-use or
	 * cached-in-reserve) to be freed.  See \p FreeAllCached().
	 */
    cudaError_t SetMaxCachedBytes(size_t max_cached_bytes) {
        // Lock
        mutex.lock();

        //   if (debug) _CubLog("Changing max_cached_bytes (%lld -> %lld)\n", (long long) this->max_cached_bytes, (long long) max_cached_bytes);

        this->max_cached_bytes = max_cached_bytes;

        // Unlock
        mutex.unlock();

        return cudaSuccess;
    }

    /**
	 * \brief Provides a suitable allocation of device memory for the given size on the specified device.
	 *
	 * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
	 * with which it was associated with during allocation, and it becomes available for reuse within other
	 * streams when all prior work submitted to \p active_stream has completed.
	 */

    cudaError_t pinnedAllocate(
      void **d_ptr,  ///< [out] Reference to pointer to the allocation
      size_t bytes);

    /**
	 * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator.
	 *
	 * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
	 * with which it was associated with during allocation, and it becomes available for reuse within other
	 * streams when all prior work submitted to \p active_stream has completed.
	 */
    cudaError_t pinnedFree(void *d_ptr);

    /**
	 * \brief Frees all cached device allocations on all devices
	 */
    cudaError_t FreeAllCached();

    /**
	 * \brief Destructor
	 */
    virtual ~CachingPinnedAllocator() {
        if (!skip_cleanup) FreeAllCached();
    }
};

#endif /* SRC_GPUABSTRACTIONS_CACHEDPINNEDALLOCATOR_H_ */
