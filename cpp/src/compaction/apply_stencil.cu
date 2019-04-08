/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2019 Eyal Rozenberg <eyalroz@blazingdb.com>
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

#include <cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <utilities/cuda_utils.hpp>
#include <utilities/column_utils.hpp>
#include <utilities/device_side_utilities.cuh>
#include <utilities/type_dispatcher.hpp>
#include <utilities/bit_util.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda_runtime.h>

#include <type_traits>
#include <utility>
#include <thread>

#define __fd__ __forceinline__ __device__
#define __fhd__ __forceinline__ __device__ __host__

using bit_container = uint32_t;
enum : unsigned { bits_per_container = gdf::util::size_in_bits<bit_container>() };
enum : gdf_bool { gdf_false = 0, gdf_true = 1 };

static_assert((unsigned) warp_size == bits_per_container,
    "Kernels in this TU rely on the bit container to have as many bits as there are lanes in a warp");
static_assert(sizeof(gdf_bool) == 1, "Unsupported size of the gdf_bool type - expecting one value per byte");

// TODO: The code here could really use an std::span-like class for bits (a "bit span" if you will) -
// that would make it easier to read and understand.

template <typename T>
inline constexpr bool is_power_of_2(T val) { return (val & (val-1)) == 0; } // Yes, this works

template <typename T>
inline constexpr T round_down_to_power_of_2(T val, T power_of_two)
{
    return val & ~(power_of_two - 1);
}

template <typename T>
inline constexpr T round_up_to_power_of_2(T val, T power_of_two)
{
    // TODO: Make sure this works for 0 like one might expect
    return round_down_to_power_of_2(val - 1, power_of_two) + power_of_two;
}

// ... up to here

namespace detail {

// This normalizes from gdf_bool to an actual bool, which if you just
// do in the calling code can appear slightly confusing. Or maybe not.
//
__fhd__ bool
get_stencil_value(const gdf_bool* stencil_data, gdf_size_type pos)
{
    return stencil_data[pos] != gdf_false;
}

struct uchar4_array { unsigned char uchars[4]; };

// This is a bit ugly, but hopefully helps performance
//
// TODO: Check whether it actually helps
__fhd__ bool
get_stencil_value(const uchar4_array& stencil_data_fragment, gdf_size_type pos)
{
    // Why not use stencil_data_fragment.uchars[pos]? Because
    // that would preculde stencil_data_fragment being placed
    // in registers; it'll have to be in local memory
    //
    // TODO: Consider making this all a single expression
    // with no branches
    switch(pos) {
    case 0: return stencil_data_fragment.uchars[0] != 0;
    case 1: return stencil_data_fragment.uchars[1] != 0;
    case 2: return stencil_data_fragment.uchars[2] != 0;
    case 3: return stencil_data_fragment.uchars[3] != 0;
    }
    return true; // Can't get here
}


} // namespace detail


namespace block {

template <typename T, unsigned ThreadsPerBlock>
__fd__ T sum(T x) {
    // Using cub is a bit of an overkill, in that not all threads actually
    // need the result. It's enough for a single arbitrary thread of the block
    // to have it. cub insists on the input and output types being the
    // same, which is generally too much to ask.
    //
    // Note: Cub uses classes instead of namespaces for some reason,
    // even when the instances of those classes don't have any interesting state.

    using block_reduce_faux_class = cub::BlockReduce<T, ThreadsPerBlock>;
    __shared__ typename block_reduce_faux_class::TempStorage reduction_scratch_space;

    return block_reduce_faux_class{reduction_scratch_space}.Sum(x);
}

template <typename T, bool AllWarpsNeedTheResult = false, unsigned MaxWarpsPerBlock = warp_size>
__fd__ T sum_over_some_full_warps(T x, unsigned num_participating_warps) {
    __shared__ T reduction_scratch_space[MaxWarpsPerBlock];
    auto warp_sum = warp::sum(x);
    if (lane::index() == 0) { reduction_scratch_space[warp::index()] = warp_sum; }
    __syncthreads(); // now the shared memory has all relevant warps' sums
    if (not AllWarpsNeedTheResult and warp::index() != 0) { return T{}; }
    auto other_warp_index = lane::index();
    auto other_warp_sum =
        (lane::index() < num_participating_warps) ? reduction_scratch_space[other_warp_index] : T{0};
    return warp::sum(other_warp_sum);
        // Here too, we could carry forward the fact that not all threads need the result. Perhaps
        // only the first one does?
}

} // namespace block

namespace kernels {

/**
 * Counts the number of indices at which two corresponding sequences of bits (of
 * the same length) are both set to 1.
 *
 * @note The bits are given in a sequence of multi-bit values; within each of these,
 * the order is from LSB to MSB. The last element may have some "slack" bits which
 * are ignored.
 *
 * @param block_pass_counts[out]
 *     Location for each GPU block's count of elements passing the filter defined
 *     by the (possibly nullable) stencil
 * @param stencil_data[in]
 *      A sequence of boolean values in a larger type (i.e. not a sequence of bits)
 * @param stencil_validity[in]
 *      Same as @p stencil_data - a sequence of bits
 * @param input_length[in]
 *     The number of elements for which to check @p stencil_data and
 *     @p stencil_validity
 */
template <
    typename Size,
    unsigned ThreadsPerBlock,
    unsigned SerializationFactor,
    bool     StencilNullability
>
__global__ void
compute_block_pass_counts(
    Size*                __restrict__  block_pass_counts,
    const gdf_bool*      __restrict__  stencil_data,
    const bit_container* __restrict__  stencil_validity,
    Size                               stencil_length // in elements
)
{
    Size thread_pass_count { 0 };
    Size elements_scanned { 0 }; // DEBUG

    // TODO: This is sub-optimal, since every validity read is of a single native word per warp,
    // while 8 would be the minimum for proper coalescing.
    auto thread_action = [&](Size pos) {
        auto stencil_element_is_valid = StencilNullability ? gdf::util::bit_is_set(stencil_validity, pos)  : true;
        if (stencil_element_is_valid) {
            thread_pass_count += detail::get_stencil_value(stencil_data, pos);
        }
        elements_scanned++;
    };
    at_block_stride(thread_action, stencil_length, SerializationFactor);
    auto block_pass_count = block::sum<Size, ThreadsPerBlock>(thread_pass_count);
    block::execute_by_single_thread([&](){ block_pass_counts[blockIdx.x] = block_pass_count; });
}

namespace warp {

//
/**
 * @brief Make it so that the value of the i'th element in an output array
 * corresponds to the i'th bit in @p unpacked_bits - and count the number
 * of on-bits (1-bits) while doing so, per thread.
 *
 * @param unpacked_bits[in] Same values
 * @param packed_bits[in]
 * @param num_bits[in] length of @unpacked bits in `UnpackedBit` elements, and of @p packed_bits in bits
 * @return the number of zero values each thread has unpacked
 */
template <typename Size, typename UnpackedBit>
__fd__ Size
unpack_bits_and_count_zeros(
    UnpackedBit*          __restrict__  unpacked_bits,
    const bit_container*  __restrict__  packed_bits,
    Size                                num_bits
)
{
    // Potential Optimization: don't use at_warp_stride and spare some comparisons
    // and repeat work for avoiding counting one bits in the slack. Alternatively,
    // consider assuming the packed bits have a zeroed-out slack.

    auto zero_count { 0 };
    at_warp_stride(num_bits, [&](Size unpacked_pos) {
        auto bit_container = packed_bits[unpacked_pos / bits_per_container];

        auto unpacked_value = gdf::util::bit_is_set(bit_container, unpacked_pos % bits_per_container);
            // it'll be either 0 or 1 - no other options
        zero_count += 1-unpacked_value;
        unpacked_bits[unpacked_pos] = unpacked_value;
    });
    return zero_count; // Each thread has a different zero count (!)
}




/**
 * @brief The latter part of stably_apply_stencil_with_block_pass_counts, and actually
 * has the same semantics, mostly.
 *
 * The differences are that:
 * 1. each warp _does not have to care_ about anything that happens in other warps and how
 *    many elements they read pass the stencil-defined filter - that's all been figured out
 *    already.
 * 2. The thread (not warp) null count is returned. Yes, that's ugly.
 *
 * @note If we were to write to the _actual_ output validity indicators, we would have had to
 * account for the first and last bit containers possibly being shared shared
 * with another warp (possibly in another block), which requires either tweaking of ranges,
 * which we don't do, or atomics, which _is_ what we do.
 *
 * @todo Possible optimization: Pass a flag indicating whether this warp's data is aligned,
 * in which case we can skip all the atomics
 *
 * @todo: Can be generalized to multiple columns (a tuple of pairs of points etc.) -
 * but need to think of ways not to avoid the benefits of `__restrict__`
 */
template <
    typename Size,
    typename ColumnElement,
    unsigned ThreadsPerBlock,
    unsigned InputElementsPerThread,
    bool     StencilNullability
//    ,
//    bool     DataNullability,
>
__device__ Size
stably_apply_stencil(
    ColumnElement*        __restrict__  warp_selected_data,
    gdf_bool*             __restrict__  unpacked_warp_selected_data_validity,
    const ColumnElement*  __restrict__  warp_input_data,
    const bit_container*  __restrict__  warp_input_data_validity,
    const gdf_bool*       __restrict__  warp_stencil_data,
    const bit_container*  __restrict__  warp_stencil_validity,
    Size                                warp_input_length,
    Size                                precomputed_warp_pass_count,
    bool                                input_is_nullable
)
{
    enum { bit_containers_per_thread = InputElementsPerThread / bits_per_container };
    enum { num_warps_per_block = ThreadsPerBlock / warp_size };
    enum { num_input_elements_covered_by_warp = warp_size * bit_containers_per_thread };

    // Optimization for extremal cases

    bool no_element_passes_the_filter = (precomputed_warp_pass_count == 0);
    if (no_element_passes_the_filter) { return 0; }

    // Possible optimization: Compute the warp null count here, and only sum up warp-level counts in the calling function.
    // Not doing this right now since we rely on cub::BlockReduction which doesn't offer this functionality; we
    // could well implement it ourselves

    bool all_elements_pass_the_filter = (precomputed_warp_pass_count == warp_input_length);
    if (all_elements_pass_the_filter) {
        ::warp::naive_copy<ColumnElement, Size>(warp_selected_data, warp_input_data, warp_input_length);
        Size thread_null_count { 0 };
        if (input_is_nullable) {
            thread_null_count = unpack_bits_and_count_zeros<Size, gdf_bool>(
                unpacked_warp_selected_data_validity, warp_input_data_validity, warp_input_length);
        }

        return thread_null_count;
    }

    // This is the typical case - some elements have passed and some haven't.
    //
    // TODO: This can use further optimization by distributing the work
    // of memory more evenly. At the moment, this could theoretically
    // be as slow as 32x the maximum throughput if only one lane in every
    // warp actually has a passing value.
    //


    // Now let's have the entire warp collaborate on each single bit container -
    // since we're finally dealing with data.

    Size thread_null_count { 0 };
    Size warp_base_output_pos { 0 };
    at_warp_stride(
        round_up_to_full_warps(warp_input_length), // This rounding does nothing except for the last warp in the last block
        [&](Size unpacked_pos) {
            auto packed_pos = unpacked_pos / warp_size;
            bool stencil_element_is_valid = lane::is_set_in_mask(warp_stencil_validity[packed_pos]);
            auto element_passes =
                detail::get_stencil_value(warp_stencil_data, unpacked_pos) and stencil_element_is_valid;

            // Figure out the intra-warp output position

            auto warp_pass_mask = ::builtins::warp_ballot(element_passes);
            constexpr const auto exclusive = false;
            auto output_pos = warp_base_output_pos + ::warp::population_count_on_preceding_lanes<exclusive>(warp_pass_mask);
            warp_base_output_pos += builtins::population_count(warp_pass_mask);
            if (not element_passes) { return; }
                // So every previous lane with a passing element contributes one here

            auto element_value = warp_input_data[unpacked_pos];
            gdf_bool element_validity = StencilNullability ?
                gdf::util::bit_is_set(warp_input_data_validity, unpacked_pos) : gdf_false;

            // And now let's write that sucker!

            warp_selected_data[output_pos] = element_value;
            if (input_is_nullable) {
                unpacked_warp_selected_data_validity[output_pos] = element_validity;
                thread_null_count += (element_validity == gdf_false);
            }

            // Potential optimization: Track the total number of elements written, and if we
            // conclude that the remainder of the warp's input range is either all-pass or no-pass,
            // stop reading from the stencil and its validity
        }
    );
    return thread_null_count;
}

} // namespace warp


namespace block {

// TODO: Check the PTX to sure std::pair isn't messing the inlining up
// for us. If it does, we might need to change the interface
// here to having non-const references for the output.
//
// ... I
template <
    typename Size,
    unsigned ThreadsPerBlock,
    bool     StencilNullability
>
std::pair<Size, Size> __device__ get_pass_counts_for_this_warp(
    const gdf_bool*       __restrict__  stencil_data,
    const bit_container*  __restrict__  stencil_validity,
    const Size*           __restrict__  block_pass_count_prefix_sums,
    Size                                input_length
)

{
    // We're going to be doing some per-warp work; but for that, each warp needs to know
    // where its output is going - meaning that we need intra-block prefix-summed counts of
    // passing elements, for contiguous (per-warp) regions

    // The computation of the warp prefix sum is a bit ugly and should be encapsulated.
    // It's rather close to a block reduction - but not quite there

    // Now we need the per-warp counts; but we can't do it at block stride,
    // since if we do that, warps will work over non-contiguous subsequences,
    // and we'll need a lot more inter-warp interactions.

    // TODO: If we're working on a small input, and only have a single block,
    // we may want to divvy up the work differently, having each warp do less. Of course,
    // those cases should also affect the grid dimensions

    auto warp_pass_count = ::warp::count_if(input_length,
        [&stencil_data, &stencil_validity](Size pos) {
            // Potential optimization: We could have arranged to make larger reads here (e.g. a full
            // bit container for each lane, and thus 32 full bytes per lane)
            auto stencil_element_is_valid = StencilNullability ?
                gdf::util::bit_is_set<bit_container, Size>(stencil_validity, pos) : true;
            return stencil_data[pos] and stencil_element_is_valid;
        }
    );
    enum { num_warps_per_block = ThreadsPerBlock / warp_size };
    __shared__ Size warp_pass_counts[num_warps_per_block];

    // At this point, each warp holds the sum of pass counts of its contiguous subsequence
    // of the block's overall sequence of elements.

    ::warp::execute_by_single_lane([&](){ warp_pass_counts[::warp::index()] = warp_pass_count; });

    __syncthreads();

    auto other_warp_index = lane::index();
    Size other_warp_pass_count  { 0 };
    if (lane::index() < ::warp::index()) {
        other_warp_pass_count  = warp_pass_counts[other_warp_index];
    }
    Size previous_warps_pass_count = ::warp::sum(other_warp_pass_count);

    auto previous_blocks_pass_count = ::block::is_first_in_grid() ?
        0 : block_pass_count_prefix_sums[blockIdx.x];
    auto global_pass_count_before_this_warp =
        previous_blocks_pass_count + previous_warps_pass_count;
    return { warp_pass_count, global_pass_count_before_this_warp };
}

} // namespace block


/**
 * Produces the elements accepted by a stencil in an output column, and
 * their validity indication - in a byte-resolution column of booleans.
 * The relative order of passing elements in the input is preserved.
 *
 * @note The prefix "stably" in due to the presentation of order, without
 * which the block pass counts would hardly be necessary apriori. One
 * might have named the cudf API function calling this kernel
 * `stably_apply_gpu_stencil()` instead of `apply_gpu_stencil()`.
 *
 * @param selected_data
 * @param unpacked_selected_data_validity
 * @param selected_data_null_count
 * @param input_data
 * @param input_data_validity
 * @param stencil_data
 * @param stencil_validity
 * @param block_pass_count_prefix_sums
 * @param input_length
 */
template <
    typename Size,
    typename ColumnElement,
    unsigned ThreadsPerBlock,
    unsigned InputElementsPerThread,
    bool     StencilNullability
>
__global__ void
stably_apply_stencil_with_block_pass_counts(
    ColumnElement*        __restrict__  selected_data,
    gdf_bool*             __restrict__  unpacked_selected_data_validity,
    Size*                 __restrict__  selected_data_null_count,
    const ColumnElement*  __restrict__  input_data,
    const bit_container*  __restrict__  input_data_validity,
    const gdf_bool*       __restrict__  stencil_data,
    const bit_container*  __restrict__  stencil_validity,
    const Size*           __restrict__  block_pass_count_prefix_sums,
    Size                                input_length
)
{
    auto block_offset_in_input = blockIdx.x * blockDim.x * InputElementsPerThread;
    auto warp_offset_in_input = block_offset_in_input + warp_size * ::warp::index() * InputElementsPerThread;
    if (warp_offset_in_input >= input_length) {
        return;
    }
    auto num_input_elements_to_process_in_this_warp =
        builtins::minimum<Size>(warp_size * InputElementsPerThread, input_length - warp_offset_in_input);

    // TODO: I'd really like C++17's structured binding here
    auto pass_counts_pair =
        block::get_pass_counts_for_this_warp<Size, ThreadsPerBlock, StencilNullability>(
            stencil_data + warp_offset_in_input,
            stencil_validity + warp_offset_in_input / bits_per_container,
            block_pass_count_prefix_sums,
            num_input_elements_to_process_in_this_warp);
    // Possible optimization: If we know this block has no passing elements, or full passing elements, we can skip
    // the call to pass_counts_pair. The price is an extra global memory read - but we save lots and lots of
    // reads from the stencils.

    auto warp_pass_count { pass_counts_pair.first };
    auto global_passing_count_before_this_warp { pass_counts_pair.second };

    bool input_is_nullable = (input_data_validity != nullptr);

    Size thread_null_count =
        warp::stably_apply_stencil<Size, ColumnElement, ThreadsPerBlock, InputElementsPerThread, StencilNullability>(
            selected_data + global_passing_count_before_this_warp,
            input_is_nullable  ? unpacked_selected_data_validity + global_passing_count_before_this_warp : nullptr,
            input_data + warp_offset_in_input,
            input_is_nullable ? input_data_validity + (warp_offset_in_input / bits_per_container) : nullptr,
            stencil_data + warp_offset_in_input,
            StencilNullability ? stencil_validity + warp_offset_in_input / bits_per_container : nullptr,
            num_input_elements_to_process_in_this_warp,
            warp_pass_count,
            input_is_nullable );

    if (input_is_nullable) {
        auto block_selected_data_null_count { 0 };
        if (blockIdx.x + 1 < gridDim.x ) {
            block_selected_data_null_count = ::block::sum<gdf_size_type, ThreadsPerBlock>(thread_null_count);
                // Note only the first thread in the block needs this value
        }
        else {
            auto num_participating_warps =
                div_by_power_of_2_rounding_up(input_length - block_offset_in_input, InputElementsPerThread * warp_size);
            constexpr const bool not_all_warps_need_the_result { false };
            block_selected_data_null_count =
                ::block::sum_over_some_full_warps<gdf_size_type, not_all_warps_need_the_result>(
                    thread_null_count, num_participating_warps);
        }
        // TODO: Nasty hack here, should use a templated version of atomic add
        static_assert(sizeof(gdf_size_type) == sizeof(int) and std::is_signed<gdf_size_type>::value == true,
            "Was expecting gdf_size_type to be more like an int - same size and signedness");
        if (threadIdx.x == 0) {
            atomicAdd(reinterpret_cast<int*>(selected_data_null_count), block_selected_data_null_count);
        }
    }

    // Note: We should be facing the problem of different warps' subranges intersecting in terms
    // of the bit containers for the validity values in the output. How can we deal with this?
    //
    // 1. We would have like to tweak the ranges a bit, and for each warp to possibly work on
    // a few more elements at its end and a few less at the beginning - but that's not possible,
    // since that might mean going over a huge number of stencil elements to locate
    // the next or previous passing input elements. It could be possible when, in the computation
    // of passing counts, we would also keep this data at the block level (and in this phase,
    // for the warp level).
    //
    // 2. Another alternative would be to write out a full _byte_ for every bit of the validity
    // column, and compact it later. It requires that much space to be allocated, which is a
    // difficult assumption to make. But perhaps we can make it right now for the sake
    // of simplicity.
    //
    // 3. Finally, we can use atomics at the beginning and the end of each warp's subrange
    // to only set 0/1 values of the appropriate bits in the relevant bit containers. This
    // will work with no assumptions, but would require more coding and more debugging,
    // and the performance penality of the atomic may be significant (need to check).
}


namespace warp {

template <typename Size, typename UnpackedBit>
__fd__ void
pack_a_quantum_into_bits(
    bit_container*      __restrict__  packed_bits,
    const UnpackedBit*  __restrict__  unpacked_values,
    Size                              num_values // no more than warp_size * bits per bit container
)
{
    // Remember that warp_size == bits_per_container; this duality is used heavily here,
    // so that sometimes we refer to the same value in different capacities. Specifically,
    // we have the same number of lanes both for a full write and for packing a single
    // bit container with balloting

    auto lane_output_pos = lane::index();
    bit_container lane_buffer { 0 };
    for(
        int warp_input_pos = 0;
        warp_input_pos < round_up_to_full_warps(num_values);
        warp_input_pos += bits_per_container)
    {
        auto lane_input_pos = warp_input_pos + lane::index();
        auto unpacked_value = lane_input_pos < num_values ? unpacked_values[lane_input_pos] : UnpackedBit{0};
            // Note it might be something other than 0 or 1, we don't care
        auto packed_container = ::warp::pack_bit_lane_bits(unpacked_value);
            // Note that if we've over-reached beyond the end of the input, we're going to have
            // 0 in the slack bits (which is a Good Thing).
        if (warp_input_pos / warp_size == lane_output_pos) { lane_buffer = packed_container; }
    }
    auto output_size = div_by_power_of_2_rounding_up(num_values, bits_per_container);
    if (lane_output_pos < output_size) {
        packed_bits[lane_output_pos] = lane_buffer;
    }
}

} // namespace warp

/**
 * Make it so that the i'th bit of the output is set if and only if the i'th value
 * in @p unpacked_values is non-zero.
 *
 * @note Slack bits _are_ written to the final `bit_container` when num_values is
 * not an integer multiple of the number of bits per container; their values is
 * guaranteed to be `0`.
 *
 * @param packed_bits[in]
 *     A sequence of bits so that, within each container, their order is LSB to MSB.
 * @param unpacked_values[in]
 *     A sequence of numeric values (typically bytes)
 * @param num_values The length of each of the sequences - the first in bits, the second
 *     in UnpackedBit's (i.e. @p packed_bits has ceil(num_values/32) `bit_container`'s.
 *
 */
template <typename Size, typename UnpackedBit>
__global__ void
pack_into_bits(
    bit_container*      __restrict__  packed_bits,
    const UnpackedBit*  __restrict__  unpacked_values,
    Size                              num_values
)
{
    // Note: There's an "inherent" serialization factor here, of warp_size = bits_per_container
    // which is reflected in the grid config, but can't really be changed as it is baked
    // in to how this is coded: 32 threads read 32 unpacked bits each, and do
    // essentially a transpose of the square matrix of "bits", then write it out

    static_assert(bits_per_container == 32, "Unsupported bit container type");
    static_assert(warp_size == bits_per_container,
        "The bit container must have as many bits as there are lanes in a warp");

    enum { warp_packing_quantum = bits_per_container * warp_size };
        // The number of elements to be packed with a single call to the warp
        // device-side function (which cannot be trivially broken down into multiple
        // independent function calls.

    auto block_unpacked_start_position = blockIdx.x * blockDim.x * warp_packing_quantum;
    auto warp_unpacked_start_position =
        block_unpacked_start_position + ::warp::index() * warp_packing_quantum;
    if (warp_unpacked_start_position >= num_values) {
        // warp_print("Nothing to do for this warp! quitting.");
        return;
    }
    auto warp_num_values_to_pack =
        builtins::minimum<Size>(warp_packing_quantum, num_values - warp_unpacked_start_position);

    warp::pack_a_quantum_into_bits(
        packed_bits + warp_unpacked_start_position / warp_size,
        unpacked_values + warp_unpacked_start_position,
        warp_num_values_to_pack);
}


} // namespace kernels


namespace detail {

template <typename T>
size_t get_scratch_space_size_for_cub_prefix_sum(size_t num_elements)
{
    size_t   scratch_space_size = 0;
    // Note: We assume this can't fail when only a scratch space size calculation is requested.
    // The cub documentation indicate that is the case - but only rather implicitly.
    cub::DeviceScan::ExclusiveSum<T*, T*>(nullptr, scratch_space_size, nullptr, nullptr, num_elements);
    return scratch_space_size;
}

gdf_size_type make_aligned_size(gdf_size_type size)
{
    enum : gdf_size_type { alignment_quantum = 256 };
    return round_up_to_power_of_2<gdf_size_type>(size, alignment_quantum);
}


// This class should be merely a templated lambda within gdf_apply_boolean_mask,
// but support for those is not available before C++17, while this code is
// targetting C++14
template <
    typename Size,
    unsigned ThreadsPerBlock,
    unsigned InputElementsPerThread
>
struct apply_stencil_helper {
    template <typename T>
    gdf_error operator()(
        void*                 __restrict__  selected_data,
        gdf_bool*             __restrict__  unpacked_selected_data_validity,
        gdf_size_type*        __restrict__  selected_data_null_count,
        const void*           __restrict__  input_data,
        const bit_container*  __restrict__  input_data_validity,
        const gdf_bool*       __restrict__  stencil,
        const bit_container*  __restrict__  stencil_validity,
        Size                                input_length,
        const Size*           __restrict__  block_pass_counts_prefix_sums,
        cudaStream_t                        stream
    )
    {
        auto selection_grid_config { cudf::util::form_naive_1d_grid(input_length, ThreadsPerBlock, InputElementsPerThread) };

        auto kernel = (stencil_validity == nullptr) ?
            kernels::stably_apply_stencil_with_block_pass_counts<Size, T, ThreadsPerBlock, InputElementsPerThread, false> :
            kernels::stably_apply_stencil_with_block_pass_counts<Size, T, ThreadsPerBlock, InputElementsPerThread, true>;

        kernel
            <<<
                selection_grid_config.num_blocks,
                ThreadsPerBlock,
                cudf::util::cuda::no_dynamic_shared_memory,
                stream
            >>>
            (
                static_cast<T*>(selected_data),
                unpacked_selected_data_validity,
                selected_data_null_count,
                static_cast<const T*>(input_data),
                input_data_validity,
                stencil,
                stencil_validity,
                block_pass_counts_prefix_sums,
                input_length
            );

        CUDA_TRY (cudaGetLastError() );
        return GDF_SUCCESS;
    }
};

} // namespace detail


/*
 * @note: all of `stencil->data`, `column->valid`, `stencil->valid` and `output->valid`
 * must have their start and end addresses of be multiples of 4, i.e. interpretable as
 * `bit_container *`
 */
gdf_error gdf_apply_boolean_mask(
    gdf_column* column,
    gdf_column* stencil,
    gdf_column* output)
{
    // The algorithm, in broad strokes:
    //
    // 1. Count the number of passing elements in several large contiguous regions of
    //    the column (e.g. one such region per kernel block).
    // 2. Perform a prefix count to obtain the starting output position for each
    //    of the block's-worth regions.
    // 3. Within each block, repeat (1.) and (2.) at the warp level with contiguous
    //    per-warp regions, so now each warp knows where its output needs to go.
    // 4. Have each warp scan its designated contiguous region and append passing
    //    elements to the output, at a final location it now can determine, and regardless
    //    of all other warps.
    //
    // The problem with this algorithm is in (4.); if we were to write directly to the
    // output valid pseudo-column, we would have to be careful not to overwrite bits
    // written by the previous and/or the next warp. Instead, we take a surrogate for
    // output->valid, in which we use full single bytes for every bit, and write there.
    // Once that's over, we continue to:
    //
    // 5. "Pack" the byte-resolution boolean column into output->valid (a bit-resolution
    //    boolean column)
    //
    // and we're done. This is not optimal - but it's easier to implement.


    // TODO: Experiment with these parameters. And does it really help for them to be
    // fixed apriori? And shouldn't they depend on microarchitecture specifics?
    //
    // Note: Keep these to powers of 2 for now
    enum { threads_per_block = 256 };
    constexpr const struct {
        unsigned selection;
        unsigned pass_count_computation;
    } elements_per_thread { warp_size, warp_size };
        // TODO: Are these reasonable figures?

    using cudf::util::cuda::no_dynamic_shared_memory;

    GDF_REQUIRE(column->size == stencil->size, GDF_COLUMN_SIZE_MISMATCH);
    GDF_REQUIRE(column->dtype == output->dtype, GDF_DTYPE_MISMATCH);
    GDF_REQUIRE(not cudf::is_nullable(*column) or cudf::is_nullable(*output), GDF_VALIDITY_MISSING);
        // TODO: Should we really accept a nullable output but non-nullable input column?

#ifndef NDEBUG
    CUDA_TRY (cudaGetLastError() );
#endif

    if (column->size == 0) {
        output->size = 0;
        output->null_count = 0;
        return GDF_SUCCESS;
    }


    // Possible optimization: perform the no-nullability-in-input work
    // if the null count is known to be 0. But then, we might need to empty
    // the validity indication column of the output if it's present

    auto pass_counting_grid_config { cudf::util::form_naive_1d_grid(column->size, threads_per_block, elements_per_thread.pass_count_computation) };

    cudf::util::cuda::scoped_stream stream;

    size_t cub_scratch_space_size = detail::make_aligned_size(
        detail::get_scratch_space_size_for_cub_prefix_sum<gdf_size_type>(pass_counting_grid_config.num_blocks));
    gdf_size_type block_pass_counts_size = pass_counting_grid_config.num_blocks * sizeof(gdf_size_type);
    gdf_size_type block_pass_counts_prefix_sums_size = block_pass_counts_size;
    gdf_size_type gynormous_unpacked_output_validity_size =
        cudf::is_nullable(*column) ? detail::make_aligned_size(column->size) : 0;
    gdf_size_type output_null_count_size = sizeof(gdf_size_type);

    auto total_scratch_space_size =
        block_pass_counts_size +
        block_pass_counts_prefix_sums_size +
        cub_scratch_space_size +
        gynormous_unpacked_output_validity_size +
        output_null_count_size;

    unsigned char* scratch_space;

    CUDA_TRY ( cudaMalloc(&scratch_space, total_scratch_space_size) );
        // TODO: This will leak memory when exiting the function prematurely on error!

    auto raw_scratch_space = scratch_space;
    auto block_pass_counts = reinterpret_cast<gdf_size_type*>(raw_scratch_space);
    auto block_pass_counts_prefix_sums = block_pass_counts + pass_counting_grid_config.num_blocks;
    auto cub_scratch_space = raw_scratch_space + (block_pass_counts_size + block_pass_counts_prefix_sums_size);
    auto gynormous_unpacked_output_validity = reinterpret_cast<gdf_bool*>(cub_scratch_space + cub_scratch_space_size);
        // One of our kernels will write a non-bit-packed version of output->valid into this scratch buffer, which
        // will later be packed into output->valid itself. This is highly sub-optimal (especially w.r.t. the amount of
        // scratch space required), but was easier and quicker to implement.
        //
        // Potential optimization - just write directly into output->valid - but make sure you avoid race conditions
        // with multiple warps or even blocks writing to the same place in device memory.
    auto output_null_count = reinterpret_cast<gdf_size_type*>(gynormous_unpacked_output_validity + gynormous_unpacked_output_validity_size);

    CUDA_TRY ( cudaMemsetAsync(output_null_count, 0, output_null_count_size, stream) );

    // We're applying the stencil in several phases - since grid-level synchronization
    // is required (and because of a compromise in the implementation of writing out
    // the bit containers of the validity pseudo-column);
    // but its important that each block of the first grid correspond to the same
    // block in the third phase grid, in that they look at the same element indices in
    // the column.
    //
    // Potential optimization: Instead of writing block-level pass counts, write warp-level
    // _and_ block-level pass counts. This will preclude the recomputation of the warp-level
    // counts in the second kernel and allow warps in the second kernel to start "working" without
    // any initial inter-warp synchronization.

    auto block_pass_counts_kernel = cudf::is_nullable(*stencil) ?
        kernels::compute_block_pass_counts<gdf_size_type, threads_per_block, elements_per_thread.pass_count_computation, true> :
        kernels::compute_block_pass_counts<gdf_size_type, threads_per_block, elements_per_thread.pass_count_computation, false>;

    block_pass_counts_kernel
        <<<
            pass_counting_grid_config.num_blocks,
            threads_per_block,
            no_dynamic_shared_memory,
            stream
        >>>
        (
            block_pass_counts,
            static_cast<const gdf_bool*>(stencil->data),
            reinterpret_cast<const bit_container*>(stencil->valid),
            column->size
        );

    if (pass_counting_grid_config.num_blocks > 1) {
        cub::DeviceScan::ExclusiveSum<const gdf_size_type*, gdf_size_type*>(
            cub_scratch_space,
            cub_scratch_space_size,
            block_pass_counts,
            block_pass_counts_prefix_sums,
            pass_counting_grid_config.num_blocks,
            stream);

        // TODO: Possible optimization: if there are few blocks, consider avoiding
        // the extra kernel and computing the prefix sum using a block-wise reduction
        // in each block of the next (and last) phase
    }

    auto result = cudf::type_dispatcher(
        column->dtype,
        detail::apply_stencil_helper<gdf_size_type, threads_per_block, elements_per_thread.selection>{},
        output->data,
        gynormous_unpacked_output_validity,
        output_null_count,
        column->data,
        reinterpret_cast<const bit_container*>(column->valid),
        static_cast<const gdf_bool*>(stencil->data),
        reinterpret_cast<const bit_container*>(stencil->valid),
        column->size,
        block_pass_counts_prefix_sums,
        static_cast<cudaStream_t>(stream)
    );
    if (result != GDF_SUCCESS) { return result; }

    // Obtain the output size using the block sums

    gdf_size_type passing_elements_before_last_block;
    gdf_size_type passing_elements_in_last_block;
    if ( pass_counting_grid_config.num_blocks > 1 ) {
        CUDA_TRY( cudf::util::cuda::copy_single_value<gdf_size_type>(
            passing_elements_before_last_block,
            block_pass_counts_prefix_sums[pass_counting_grid_config.num_blocks - 1],
            stream
        ) );
    }
    else {
        passing_elements_before_last_block = 0;
    }

    CUDA_TRY( cudf::util::cuda::copy_single_value<gdf_size_type>(
        passing_elements_in_last_block,
        block_pass_counts[pass_counting_grid_config.num_blocks - 1],
        stream
    ) );
    output->size = passing_elements_before_last_block + passing_elements_in_last_block;

    if (cudf::is_nullable(*column)) {
        CUDA_TRY (cudaMemcpyAsync(&output->null_count, output_null_count, sizeof(gdf_size_type), cudaMemcpyDeviceToHost, stream) );
        if (output->size > 0) {
            auto pack_grid_config = cudf::util::form_naive_1d_grid(
                round_up_to_power_of_2<gdf_size_type>(output->size, bits_per_container),
                threads_per_block,
                bits_per_container);
            kernels::pack_into_bits<gdf_size_type, gdf_bool>
                <<<
                    pack_grid_config.num_blocks,
                    threads_per_block,
                    no_dynamic_shared_memory,
                    stream
                >>>
                (
                    reinterpret_cast<bit_container*>(output->valid),
                    gynormous_unpacked_output_validity,
                    output->size
                );
            CUDA_TRY( cudaStreamSynchronize(stream));
            CUDA_TRY (cudaGetLastError() );
        }
    }
    else if (cudf::is_nullable(*output)) {
        CUDA_TRY( cudaMemsetAsync(output->valid, ~gdf_valid_type{0}, gdf_valid_allocation_size(output->size), stream) );
    }

    CUDA_TRY( cudaStreamSynchronize(stream));
    CUDA_TRY( cudaFree(scratch_space) );
        // Note that if we've failed failed before reaching this point, memory leaks
    CUDA_TRY (cudaGetLastError() );
	return GDF_SUCCESS;
} 

#undef __fd__
#undef __fhd__
