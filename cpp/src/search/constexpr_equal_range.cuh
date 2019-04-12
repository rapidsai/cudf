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


//===-------------------------- ---------------------------------===//
//
// The source code in this file was adapted from the LLVM libc++,
// part of the LLVM Project, which is available under the Apache License v2.0
// with LLVM Exceptions. See https://llvm.org/LICENSE.txt for license
// information. The original source code appears at:
//
// https://github.com/llvm/llvm-project/blob/master/libcxx/include/algorithm
//
// (commit ID: c8879ab2fd9565120e835abc84e0ad9f66bfd021 )
//
//===----------------------------------------------------------------------===//

#include <utility>
#include <type_traits>
#include <iterator>

#ifndef __NVCC__
#ifndef __host__
#define __host__
#define __device__
#endif
#endif

namespace cudf {

namespace util {


template <class ForwardIterator, class T>
constexpr ForwardIterator
lower_bound(ForwardIterator first, ForwardIterator last, const T& value)
{
    using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
    difference_type len = // std::distance(first, last);  // <-- That's not constexpr before C++17!
        last - first;
    while (len != 0) {
        difference_type l2 = // std::half_positive(len);
            static_cast<difference_type>(static_cast<typename std::make_unsigned<difference_type>::type>(len) / 2);
        ForwardIterator m = first;
        // std::advance(m, l2);
        m += l2;
        if (*m < value) {
            first = ++m;
            len -= l2 + 1;
        }
        else {
            len = l2;
        }
    }
    return first;
}

template <class ForwardIterator, class T>
constexpr __host__ __device__ ForwardIterator
upper_bound(ForwardIterator first, ForwardIterator last, const T& value)
{
    using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
    difference_type len = // std::distance(first, last);  // <-- That's not constexpr before C++17!
        last - first;
    while (len != 0) {
        difference_type l2 = // std::half_positive(len);
            static_cast<difference_type>(static_cast<typename std::make_unsigned<difference_type>::type>(len) / 2);
        ForwardIterator m = first;
        // std ::advance(m, l2);
        m += l2;
        if (value < *m)  // comp(value, *m)
        {
            len = l2;
        }
        else {
            first = ++m;
            len -= l2 + 1;
        }
    }
    return first;
}

template <class ForwardIterator, class T>
constexpr __host__ __device__  std::pair<ForwardIterator, ForwardIterator>
equal_range(ForwardIterator first, ForwardIterator last, const T& value)
{
    using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
    difference_type len = // std::distance(first, last);  // <-- That's not constexpr before C++17!
        last - first;
    while (len != 0) {
        difference_type l2 = // std::half_positive(len);
            static_cast<difference_type>(static_cast<typename std::make_unsigned<difference_type>::type>(len) / 2);
        ForwardIterator m = first;
        // std::advance(m, l2);  // <-- That's not constexpr before C++17!
        m += l2;
        if (*m < value) {
            first = ++m;
            len -= l2 + 1;
        }
        else if (value < *m) // comp(value, *m)
        {
            last = m;
            len = l2;
        }
        else {
            ForwardIterator mp1 = m;
            return std::pair<ForwardIterator, ForwardIterator>(
                lower_bound<ForwardIterator, T>(first, m, value),
                upper_bound<ForwardIterator, T>(++mp1, last, value)
                // Note, however, that equal_range uses a call to upper_bound and a call to lower_bound,
                // in sequence, which is less optimal than combining the two calls - especially on a GPU
                // where waiting for a memory read is quite expensive.
            );
        }
    }
    return std::pair<ForwardIterator, ForwardIterator>(first, first);
}

} // namespace util

} // namespace cudf

#ifndef __NVCC__
#undef __host__
#undef __device__
#endif
