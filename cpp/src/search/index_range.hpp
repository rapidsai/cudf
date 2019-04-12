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

#include <type_traits>
#include <iterator>


#ifdef NVCC
#define __fhd__ __forceinline__ __host__ __device__
#define __hd__  __host__ __device__
#else
#define __fhd__ inline
#define __hd__
#endif

namespace cudf {

namespace util {

/**
 * @brief A super-simple index range - but meeting the stand library's container requirements,
 * sort of (no real pointers or references since it's just a facade).
 *
 * @note Boost has an irange class, but let's not go there.
 */
template <typename I>
struct index_range {
    using value_type = I;
    using reference = I;  // No references!
    using const_reference = I;  // No references!
    using difference_type = std::make_signed_t<I>;
    using size_type = I;

    I start;  // first element which _may_ be the one we're after (i.e.
                          // the first greater or the first greater-or-equal); before
                          // this one, all elements are lower / lower-or-equal than our pivot

    I end_;    // Arrgh, can't use `end` as a data member name, it's taken

    constexpr __fhd__ I     middle()      const { return start + (end_ - start) / 2; }
    constexpr __fhd__ I     last()        const { return end_ - 1; }
    constexpr __fhd__ I     length()      const { return end_ - start; } // TODO: What if end_ < start?
    constexpr __fhd__ bool  is_empty()    const { return length() <= 0; }

    // Why does the standard library like ambiguous phrases for method names?!
    constexpr __fhd__ bool  empty()       const { return is_empty(); }
    constexpr __fhd__ bool  is_singular() const { return length() == 1; }

    constexpr __fhd__ void  drop_lower_half() { start = middle(); }
    constexpr __fhd__ void  drop_upper_half() { end_   = middle(); }

    constexpr __fhd__ size_type max_size() const { return std::numeric_limits<difference_type>::max(); }

    // named constructor idioms

    static __fhd__ index_range constexpr trivial_empty() { return { 0, 0 }; }
    static __fhd__ index_range constexpr singular(I index) { return { index, index+1 }; }
    static __fhd__ index_range constexpr empty_at(I index) { return { index, index }; }

    __fhd__ void swap(index_range& other) {
        index_range tmp = other;
        other = *this;
        *this = tmp;
    }

    struct iterator {
    public:
        const index_range<I> range_;
        I pos_;

    public:
        constexpr __fhd__ iterator(index_range<I> range, I pos) : range_(range), pos_(pos) { };
        constexpr __fhd__ iterator(const iterator& other) = default;
        constexpr __fhd__ iterator(iterator&& other) = default;
        constexpr __fhd__ iterator& operator++() { if (pos_ < range_.end_) pos_++; return *this;}
        constexpr __fhd__ iterator operator++(int) {iterator retval = *this; ++(*this); return retval;}
        constexpr __fhd__ bool operator==(iterator other) const {return range_ == other.range_ and pos_ == other.pos_; }
        constexpr __fhd__ bool operator!=(iterator other) const {return not (*this == other);}
        constexpr __fhd__ I operator*() const {return pos_;} // Yeah, it's const anyway

        // TODO: Consider adding operator[]'s

        // iterator traits
        using value_type = typename index_range<I>::value_type;;
        using difference_type = typename index_range<I>::difference_type;
        using pointer = I*;
        using reference = I;
        using iterator_category = std::random_access_iterator_tag;
    };
    constexpr __fhd__ iterator begin() {return iterator(this, start);}
    constexpr __fhd__ iterator end() {return iterator(this, end_);  }
    constexpr __fhd__ iterator cbegin() {return begin(); }
    constexpr __fhd__ iterator cend() {return end(); }
};

template <typename I>
constexpr __fhd__ index_range<I> swap(index_range<I>& lhs, index_range<I>& rhs)
{
    return lhs.swap(rhs);
}

template <typename I>
constexpr __fhd__ index_range<I> lower_half(index_range<I> range)
{
    return { range.start, range.middle() };
}
template <typename I>
constexpr __fhd__ index_range<I> lower_half_and_middle(index_range<I> range)
{
    return { range.start, range.middle()+1 };
}
template <typename I>

constexpr __fhd__ index_range<I> upper_half(index_range<I> range)
{
    return { range.middle(), range.end_ };
}

template <typename I>
constexpr __fhd__ index_range<I> strict_upper_half(index_range<I> range)
{
    return { range.middle() + 1, range.end_ };
}

template <typename I>
constexpr __fhd__ index_range<I> intersection(index_range<I> lhs, index_range<I> rhs)
{
    const auto& starts_first  { (lhs.start < rhs.start) ? lhs : rhs };
    const auto& starts_second { (lhs.start < rhs.start) ? rhs : lhs };

    if (starts_first.end_ <= starts_second.start) {
        return index_range<I>::trivial_empty();
    }
    return { starts_second.start, starts_first.end_ };
}

} // namespace util
} // namespace cudf



#undef __fhd__
#undef __fd__

