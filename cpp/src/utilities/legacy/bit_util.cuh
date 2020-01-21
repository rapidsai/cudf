
/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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
#pragma once

#include <utilities/legacy/cudf_utils.h>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cudf/types.h>

#include <cstdint>
#include <climits>
#include <string>

namespace cudf {
namespace util {

template <typename T>
constexpr inline std::size_t size_in_bits() { return sizeof(T) * CHAR_BIT; }

template <typename T>
constexpr inline std::size_t size_in_bits(const T&) { return size_in_bits<T>(); }

namespace detail {

template <typename BitContainer, typename Size>
constexpr CUDA_HOST_DEVICE_CALLABLE
Size intra_container_index(Size bit_index) { return bit_index % size_in_bits<BitContainer>(); }

template <typename BitContainer, typename Size>
constexpr CUDA_HOST_DEVICE_CALLABLE
Size bit_container_index(Size bit_index) { return bit_index / size_in_bits<BitContainer>(); }


} // namespace detail


template <typename BitContainer, typename Size>
constexpr CUDA_HOST_DEVICE_CALLABLE
void turn_bit_on(BitContainer* bits, Size bit_index)
{
    auto container_index = detail::bit_container_index<BitContainer, Size>(bit_index);
    auto intra_container_index = detail::intra_container_index<BitContainer, Size>(bit_index);
    bits[container_index] |= (BitContainer{1} << intra_container_index);
}

template <typename BitContainer, typename Size>
constexpr CUDA_HOST_DEVICE_CALLABLE
void turn_bit_off(BitContainer* bits, Size bit_index)
{
    auto container_index = detail::bit_container_index<BitContainer, Size>(bit_index);
    auto intra_container_index = detail::intra_container_index<BitContainer, Size>(bit_index);
    bits[container_index] &= ~((BitContainer{1} << intra_container_index));
}

/**
 * Checks if a bit is set within a bit-container, in which the bits
 * are ordered LSB to MSB
 *
 * @param bits[in] a bit container
 * @param bit_index[in] index within the sequence of bits in the container
 * @return true iff the bit is set
 */
template <typename BitContainer, typename Size>
constexpr CUDA_HOST_DEVICE_CALLABLE bool bit_is_set(const BitContainer& bit_container, Size bit_index)
{
    auto intra_container_index = detail::intra_container_index<BitContainer, Size>(bit_index);
    return bit_container & (BitContainer{1} << intra_container_index);
}

/**
 * Checks if a bit is set in a sequence of bits in container types,
 * such that within each container the bits are ordered LSB to MSB
 *
 * @param bits[in] pointer to the beginning of the sequence of bits
 * @param bit_index[in] index to bit check in the sequence
 * @return true iff the bit is set
 */
template <typename BitContainer, typename Size>
constexpr CUDA_HOST_DEVICE_CALLABLE bool bit_is_set(const BitContainer* bits, Size bit_index)
{
    auto container_index = detail::bit_container_index<BitContainer, Size>(bit_index);
    return bit_is_set<BitContainer, Size>(bits[container_index], bit_index);
}

template <typename BitContainer, typename Size>
inline constexpr cudf::size_type packed_bit_sequence_size_in_bytes (Size num_bits) {
    return cudf::util::div_rounding_up_safe<Size>(num_bits, size_in_bits<BitContainer>());
}


} // namespace util
} // namespace cudf
