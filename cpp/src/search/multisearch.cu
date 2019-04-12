/*
 * Copyright 2019 BlazingDB, Inc.
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


#include "column_sequence_utils.hpp"
#include "constexpr_equal_range.cuh"
#include "index_range.hpp"

#include <cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/bit_util.cuh>
#include <utilities/error_utils.hpp>
#include <utilities/column_utils.hpp>
#include <utilities/integer_utils.hpp>
#include <utilities/cuda_utils.hpp>
#include <utilities/type_dispatcher.hpp>
#include <search/fill.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <type_traits>

#ifdef __NVCC__
#define __fhd__ __forceinline__ __host__ __device__
#define __fd__ __forceinline__ __device__
#else
#define __fhd__ inline
#define __fd__ inline
#endif

template <typename I> __fhd__ constexpr I logical_xor (I x, I y) noexcept { return (    x and not y) or (not x and     y); }
template <typename I> __fhd__ constexpr I logical_nxor(I x, I y) noexcept { return (    x and     y) or (not x and not y); }

namespace warp {
__fd__ auto ballot(int pred) {
    enum : unsigned { full_warp_mask = 0xFFFFFFFFU };
    return __ballot_sync(full_warp_mask, pred);
}

} // namespace warp

namespace kernels {

using index_range = cudf::util::index_range<gdf_size_type>;

namespace thread {


// TODO: Would be sped up/shortened by templating the flag.
/**
 * Find the subrange of NULL values within a contiguous subcolumn,
 * provided that NULLs appear either all at the beginning or all
 * at the end of the column.
 *
 * @param range We limit our search to the subcolumn this defines;
 *     we cannot take iteraotrs since those are not defined for
 *     bit-resolution data in a standard manner.
 * @param validity_pseudocolumn The pseudocolumn of validity bits / null indicators;
 *     we don't search all of it, only the designated @p range of indices
 * @param nulls_appear_before_values If true, we may assume
 *    @p validity_pseudocolumn begins with some NULLs (or none at all), and
 *    at some point the NULLs end and only non-null values appear later.
 *     When false, it's the same but for the _end_ of the column, i.e.
 *     up to some point there are no nulls and then only nulls
 *
 * @note With a bit span class where the bit sequence does not
 * necessarily begin at a bit container boundary, we could have
 * just used @ref std::equal_range instead of this function.
 *
 * @return a subrange of @p range, whose values are all NULL (invalid),
 * and with no NULLs outside it within @p range. It is guaranteed that
 * if @p nulls_appear_before_values , the range will start at range.start_,
 * and if not @p nulls_appear_before_values, the range will end at range.end_ -
 * even for empty ranges.
 */
__device__ index_range
nullness_subrange(
    index_range             range,
    const gdf_valid_type *  validity_pseudocolumn,
    bool                    nulls_appear_before_values
)
{
    if (validity_pseudocolumn == nullptr) {
        return nulls_appear_before_values ?
            index_range{range.start, range.start} :
            index_range{range.end_, range.end_};
    }

    bool first_is_valid = cudf::util::bit_is_set(validity_pseudocolumn, range.start);
    bool last_is_valid = cudf::util::bit_is_set(validity_pseudocolumn, range.last());

    if (first_is_valid and last_is_valid) {
        return index_range::empty_at(
            nulls_appear_before_values ? range.start : range.end_
        );
    }

    // At this point we know there's a transition from valid to invalid or vice-versa
    // somewhere in the column, and we need to locate that point.

    assert(logical_nxor(first_is_valid, nulls_appear_before_values));

    index_range transition_range { range };
        // Invariant: The first element of this range is before the transition, and the last element
        // is after it. Which also means transition_range has at least 2 elements.

    // Contract the transition search range using bisections

    while (transition_range.length() > 2) {
        auto middle_index = transition_range.middle();
        auto middle_element_is_valid = cudf::util::bit_is_set(validity_pseudocolumn, middle_index);
        auto middle_element_is_before_transition =
            logical_nxor(nulls_appear_before_values, middle_element_is_valid);
        transition_range = middle_element_is_before_transition ?
            lower_half_and_middle(transition_range) :
            upper_half(transition_range);
    }
    assert(transition_range.length() == 2);
    return nulls_appear_before_values ?
        index_range{ range.start, transition_range.last()} :
        index_range{ transition_range.last(), range.end_};
}


/**
 * @brief Try to continue narrowing the range of possible locations of the haystack
 * element we're looking for (the first one greater than / greater-or-equal than
 * the needle) using a single column.
 *
 * @note when this is run, we must have noticed that haystack records
 * before @p equality_range.start are less-or-equal than / less than the
 * needle - but we were using previous columns (in the haystack and needle) for that;
 * similarly, records at @p equality_range.end_ - 1 and later are greater than
 * the needle. It must be the case that the rest of the records in the range are
 * identical to the needle up until this column.
 *
 * @note We assume the nullability matches, i.e. if the needle is null, then
 * the column is nullable.
 *
 *
 * @param[in] equality_range
 *     the range of column indices whose elements we want to check for equality
 *     with the needle element. Guaranteed to have more than a single element
 *     when the function is called!
 *
 * @return The subrange of @p equality_range containing all elements which
 * are order-equivalent to the needle, and if FindFirstGreater is true, also the
 * first element coming after the needle, if one exists.
 * po * first
 */
template <typename E>
__device__ index_range  equality_subrange(
    index_range                           range,
    const E*               __restrict__   column,
    const gdf_valid_type * __restrict__   column_validity,
    const E&               __restrict__   value,
    bool                                  value_is_valid,
    bool                                  nulls_appear_before_values
)
{
    assert(not range.is_empty() and "Range contraction should not be attempted for an empty range");

    auto nullness_subrange_ = nullness_subrange(
        range,
        column_validity,
        nulls_appear_before_values);

    if (not value_is_valid) {
        return nullness_subrange_;
    }
    auto valid_elements_subrange = nulls_appear_before_values ?
        index_range{nullness_subrange_.end_, range.end_} :
        index_range{range.start, nullness_subrange_.start};

    // At this point we know we're looking for a valid value. We don't need to reinvent the wheel
    // on this one - there's a standard library function to do just this: std::equal_range().
    // Unfortunately, it's not constexpr until C++17, so we had to make a sort of a constexpr
    // copy of it

    std::pair<const E*, const E*> result =
        cudf::util::equal_range(
            column + valid_elements_subrange.start,
            column + valid_elements_subrange.end_,
            value
        );

    return index_range {
        static_cast<gdf_size_type>(result.first - column),
        static_cast<gdf_size_type>(result.second - column)
    };
}

namespace detail {

/**
 * @brief A wrapper around @ref equality_subrange which obtains the
 * (typed) needle record element.
 *
 * With C++17, this would have been a simple lambda within the `search` function.
 * Although... we could have a version of type_dispatcher which imbues certain
 * arguments with the column element type, e.g. based on some tag.
 */
struct equality_subrange_helper {

    // Note: For some reason, we get an error message if we drop the __host__
    // specifier - even though this function is never called by host-side code.
    template <typename E>
    __host__ __device__ index_range  operator()(
        index_range                           range,
        const void*            __restrict__   column,
        const gdf_valid_type * __restrict__   column_validity,
        const void*            __restrict__   values,
        gdf_index_type                        value_index,
        bool                                  value_is_valid,
        bool                                  nulls_appear_before_values
    )
    {
        return equality_subrange(
            range,
            static_cast<const E*>(column),
            column_validity,
            static_cast<const E*>(values)[value_index],
            value_is_valid,
            nulls_appear_before_values);
    }
};


} // namespace detail

struct search_result {
    bool          found;
    gdf_size_type pos;
};

search_result
__device__
search(
    const void            * __restrict__ const * __restrict__  sorted_haystack_data,
    const gdf_valid_type  * __restrict__ const * __restrict__  sorted_haystack_validities,
    const void            * __restrict__ const * __restrict__  needle_data,
    const gdf_valid_type  * __restrict__ const * __restrict__  needle_validities,
    const gdf_dtype       * __restrict__                       column_data_types,
    gdf_column_index_type                                      num_columns,
    gdf_size_type                                              haystack_length,
    gdf_size_type                                              needle_index,
    bool                                                       nulls_appear_before_values,
    bool                                                       find_first_greater
)
{
    index_range potential_equality_range { 0, haystack_length };

    for(gdf_column_index_type column_index = 0; column_index < num_columns; column_index++)
    {
        if (potential_equality_range.is_empty()) { break; }

        auto needle_column_validities = needle_validities[column_index];

        bool needle_record_element_is_valid =
            needle_column_validities == nullptr ?
                true :
                cudf::util::bit_is_set(needle_column_validities, needle_index);

        // Note:
        // We can't focus on just the bottom or the top of the range of equality
        // for the next column's needle value, since we cannot know which part
        // of the equality range will be relevant after further columns are taken
        // into consideration. So - we have to to get the whole range.
        potential_equality_range = cudf::type_dispatcher(
            column_data_types[column_index],
            detail::equality_subrange_helper{},
            // arguments
            potential_equality_range,
            sorted_haystack_data[column_index],
            sorted_haystack_validities[column_index],
            needle_data[column_index],
            needle_index,
            needle_record_element_is_valid,
            nulls_appear_before_values
        );
    }

    index_range equality_range { potential_equality_range };
        // The range of indices at which all column records are equal to the needle

    if (find_first_greater) {
        auto found = equality_range.end_ < haystack_length;
        return { found, equality_range.end_ };
    }
    if (equality_range.is_empty()) {
        return { equality_range.end_ < haystack_length, equality_range.end_};
    }
    return { true, equality_range.start};
}


} // namespace thread

/**
 * A quick and (somewhat) dirty implementation of a multisearch (i.e.
 * a search for multiple needles within a single haystack), in which
 * each element in the haystack is a _record_ of values in several columns;
 * and the record's sequence of types is known only at run-time.
 *
 * @note It is effectively impossible to template over the sequence of
 * element types, as there is a nigh-infinite set of combinations of
 * column types we could use, and they are not known in compile-time, but
 * rather at run time.
 *
 * @note For a description of the parameters, see @ref ::gdf_multisearch() .
 *
 *
 * @todo we should have a completely separate implementation for the case
 * where all columns are known not to be null, in which case the comparisons
 * are faster, we have less parameters, less register use, etc.
 */
void
__global__ multisearch(
    gdf_size_type         * __restrict__                       result,
    gdf_valid_type        * __restrict__                       result_validities,
    const void            * __restrict__ const * __restrict__  sorted_haystack_data,
    const gdf_valid_type  * __restrict__ const * __restrict__  sorted_haystack_validities,
    const void            * __restrict__ const * __restrict__  needle_data,
    const gdf_valid_type  * __restrict__ const * __restrict__  needle_validities,
    const gdf_dtype       * __restrict__                       column_data_types,
    gdf_column_index_type                                      num_columns,
    gdf_size_type                                              num_needles,
    gdf_size_type                                              haystack_length,
    bool                                                       nulls_appear_before_values,
    bool                                                       find_first_greater,
    bool                                                       use_haystack_length_for_not_found
)
{
    assert(haystack_length < std::numeric_limits<gdf_size_type>::max() and "The extremal gdf_size_type column size is not supported");
    assert(haystack_length > 0 and "Don't call this kernel with an empty haystack");

    // TODO: Consider placing the thread's needle validities in (spilled) local memory, or in shared memory.

    auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    auto thread_needle_index = global_thread_index;
    // Ha ha, see what I did just there? thread-needle? Get it? Get it? :-)

    if (thread_needle_index  >= num_needles) { return; }

    auto search_result = thread::search(
        sorted_haystack_data, sorted_haystack_validities,
        needle_data, needle_validities,
        column_data_types,
        num_columns,
        haystack_length,
        thread_needle_index,
        nulls_appear_before_values,
        find_first_greater);

    if (search_result.found) {
        result[thread_needle_index] = search_result.pos;
    }
    else {
        if (use_haystack_length_for_not_found) {
            result[thread_needle_index] = haystack_length;
        }
    }

    if (result_validities == nullptr) { return; }
    auto lane_validity_bit = (search_result.found or use_haystack_length_for_not_found);
    enum : unsigned { full_warp_mask = 0xFFFFFFFFU };
    auto warp_validity_bit_container = ::warp::ballot(lane_validity_bit);

    auto am_first_lane_in_warp = (threadIdx.x % warpSize == 0);
    if (am_first_lane_in_warp) {
        // We rely here on the assumption that the allocated space for valid pseudo-columns
        // has a 4-byte alignment.
        // TODO: Should we, instead, have every 8th lane write a byte?
        auto bit_container_pos = cudf::util::detail::bit_container_index<uint32_t, gdf_size_type>(thread_needle_index);
        reinterpret_cast<uint32_t*>(result_validities)[bit_container_pos] = warp_validity_bit_container;
    }
}

} // namespace kernels

namespace detail {

void validate_inputs(
    const gdf_column *          result,
    const gdf_column * const *  sorted_haystack,
    const gdf_column * const *  needles,
    gdf_num_columns_type        num_columns,
    bool                        use_haystack_length_for_not_found
)
{
    using namespace cudf;
    // I'd cast the **'s into std::span's / gsl::span's if I could!

    if (not (num_columns > 0)) {
        throw std::invalid_argument("No columns to search");
    }

    cudf::validate_all(sorted_haystack, num_columns);
    cudf::validate_all(needles,         num_columns);

    if (not has_uniform_column_sizes(sorted_haystack, num_columns)) {
        throw std::invalid_argument("Haystack columns must all have the same size (number of elements)");
    }
    if (not has_uniform_column_sizes(needles, num_columns)) {
        throw std::invalid_argument("Haystack columns must all have the same size (number of elements)");
    }
    if (not have_matching_types(sorted_haystack, needles, num_columns)) {
        throw std::invalid_argument("Haystack columns must all have the same size (number of elements)");
    }

    cudf::validate(result);
    if (result->dtype != GDF_SIZE_TYPE) {
        throw std::invalid_argument("The result column will hold record indices, so its element type must be GDF_SIZE_TYPE");
    }
    if (not is_nullable(*result) and not use_haystack_length_for_not_found) {
        throw std::invalid_argument("The result column will hold NULLs when no appropriate element is found - and so cannot be non-nullable");
    }
}

/**
 * @brief A wrapper around one of the rmm::device_vector containers, making
 * it clear that it's a big deal - copying a bunch of data from the host to
 * the GPU device and not something trivial.
 *
 * @note Strange that we don't have to specify the device here.
 */
template <typename Container>
rmm::device_vector<typename Container::value_type>
inline make_device_side_copy_of(const Container& container)
{
    return { container };
}

/**
 * @brief bunch together a certain attribute for each of several columns,
 * and make the result available in contiguous GPU device global memory.
 *
 * @note Strange that we don't have to specify the device here.
 *
 * @todo Do we really need the __restrict__'s here? If so, it's
 * only because of potential aliasing due to F. I wonder how compilers
 * behave on this one.
 *
 * @param columns
 * @param num_columns
 * @param attribute_getter
 * @return
 */
template <typename F>
auto make_on_device_attributes(
    gdf_column * __restrict__ * __restrict__  columns,
    gdf_column_index_type                     num_columns,
    F                                         attribute_getter)
{
#if __cplusplus < 201703L
    using attribute_value_type = std::result_of_t<F(const gdf_column&)>;
#else
    using attribute_value_type = std::invoke_result_t<F, gdf_size_type>;
#endif

    // TODO: This is super slow. It can be sped up by:
    // 1. Using an std::array with a fixed (but not so small) size
    //    for handling the common cases of not-so-many columns (say, less
    //    than 32 just to throw a number around; that shouldn't tax the
    //    stack too much.
    // 2. Better yet: Using preallocated pinned host memory.
    // 3. Not having multiple calls to this function, but rather a single
    //    call with a "multi-getter", or with multiple getters that,
    //    hopefully, get optimized away.

    std::vector<attribute_value_type> host_side_attributes;
    host_side_attributes.reserve(num_columns);

    for(gdf_column_index_type i = 0; i < num_columns; i++)
    {
        auto attribute = attribute_getter(*(columns[i]));
        host_side_attributes.emplace_back(attribute);
    }

    return detail::make_device_side_copy_of(host_side_attributes);
}

} // namespace detail

/*
 * A quick (as opposed to optimal) and somewhat-dirty implementation.
 *
 * Some Possible optimizations:
 * - Split work into two kernels: The first determines the transition point
 *   from null to non-null (or vice-versa) in each of the haystack columns, while the second
 *   performs the multisearch with that knowledge available.
 * - Sort the needles, fully or partially. Or - assume they're sorted
 * - Don't perform the search for "duplicate needles" - do the search for each distinct needle,
 *   then replicate the result as necessary. Or - assume there are no duplicate needles.
 * - Create a shared memory size's "sketch" of the data first, then have each block start
 *   by loading that from global memory, searching that, then finalizing the search using the
 *   actual data in global memory.
 * - Use optional-like structures for data+nullness indication, and zip iterators
 *   instead of pairs-of-pointers... provided that doesn't impact performance much. Actually,
 *   that would be more of an aesthetic than a performance improvement; and it's not
 *   relevant only for this API function.
 * - Count the nulls within the search kernel(s)
 *
 */
gdf_error
gdf_multisearch(
    gdf_column *           results,
    gdf_column **          sorted_haystack,
    gdf_column **          needles,
    gdf_column_index_type  num_columns,
    bool                   find_first_greater,
    bool                   nulls_appear_before_values,
    bool                   use_haystack_length_for_not_found
)
{
    detail::validate_inputs(results, sorted_haystack, needles, num_columns, use_haystack_length_for_not_found);

    gdf_size_type num_needles = needles[0]->size;

    if (num_needles == 0) { return GDF_SUCCESS; }

    cudf::util::cuda::scoped_stream stream;

    gdf_size_type haystack_length = sorted_haystack[0]->size;

    if (haystack_length == 0) {
        // See how the following two lines should really have been something like an optional?
        auto fill_value = use_haystack_length_for_not_found ? haystack_length : gdf_size_type{};
        auto fill_with_nulls = not use_haystack_length_for_not_found;
        cudf::fill(*results, stream, fill_value, fill_with_nulls);
        CUDA_TRY( cudaStreamSynchronize(stream) );
        return GDF_SUCCESS;
    }

    using detail::make_on_device_attributes;
    auto sorted_haystack_data       = make_on_device_attributes(sorted_haystack, num_columns, [](const gdf_column& c) { return c.data;  });
    auto sorted_haystack_validities = make_on_device_attributes(sorted_haystack, num_columns, [](const gdf_column& c) { return c.valid; });
    auto column_data_types          = make_on_device_attributes(sorted_haystack, num_columns, [](const gdf_column& c) { return c.dtype; });
        // Recall these types apply both to the haystack and the needles
    auto needle_data                = make_on_device_attributes(needles,         num_columns, [](const gdf_column& c) { return c.data;  });
    auto needle_validities          = make_on_device_attributes(needles,         num_columns, [](const gdf_column& c) { return c.valid; });

    // Note: With C++17, we could have made these lambda's take a const auto& c


    // TODO: What about the extra type info? For now, ignoring it, though we really shouldn't

    // TODO: Nearly-sort the needles, so that each warp gets to act on needles that are relatively close together.
    // Perhaps even space-out the needles so that no warp has to work on highly disparate needles (which would make it finish slower).
    // Of course, sorting is quite expensive, so this becomes irrelevant when |needles| approaches the same order of magnitude as |haystack|.

    enum { threads_per_block = 256 }; // TODO: This should account for information regarding the device
    cudf::util::cuda::simple_1d_grid grid_config {num_needles, threads_per_block };

    kernels::multisearch
    <<<
        grid_config.num_blocks,
        grid_config.num_threads_per_block,
        cudf::util::cuda::no_dynamic_shared_memory,
        stream
    >>>(
        cudf::get_data<gdf_size_type>(results),
        results->valid,
        sorted_haystack_data.data().get(),
        sorted_haystack_validities.data().get(),
        needle_data.data().get(),
        needle_validities.data().get(),
        column_data_types.data().get(),
        num_columns,
        num_needles,
        haystack_length,
        nulls_appear_before_values,
        find_first_greater,
        use_haystack_length_for_not_found
    );

    CUDA_TRY( cudaStreamSynchronize(stream) );

    if (use_haystack_length_for_not_found) {
        results->null_count = 0;
    }
    else {
        set_null_count(*results);
    }

    return GDF_SUCCESS;
}

#undef __fhd__
#undef __fd__
