/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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


#include <cstdlib>

namespace
{
using BYTE = uint8_t;

// number of characters in a string computed on-demand
// the _length member is initialized to this value as a place-holder
constexpr cudf::size_type UNKNOWN_STRING_LENGTH{-1};

/**
 * @brief Returns the number of bytes used to represent the provided byte.
 * This could be 0 to 4 bytes. 0 is returned for intermediate bytes within a
 * single character. For example, for the two-byte 0xC3A8 single character,
 * the first byte would return 2 and the second byte would return 0.
 *
 * @param byte Byte from an encoded character.
 * @return Number of bytes.
 */
__host__ __device__ inline cudf::size_type bytes_in_utf8_byte(BYTE byte)
{
    cudf::size_type count = 1;
    count += (int)((byte & 0xF0) == 0xF0); // 4-byte character prefix
    count += (int)((byte & 0xE0) == 0xE0); // 3-byte character prefix
    count += (int)((byte & 0xC0) == 0xC0); // 2-byte character prefix
    count -= (int)((byte & 0xC0) == 0x80); // intermediate byte
    return count;
}

/**
 * @brief Returns the number of bytes used in the provided char array by
 * searching for a null-terminator byte.
 *
 * @param str Null-terminated array of chars.
 * @return Number of bytes.
 */
__device__ inline cudf::size_type string_bytes( const char* str )
{
    if( !str )
        return 0;
    cudf::size_type bytes = 0;
    while(*str++)
        ++bytes;
    return bytes;
}

} // namespace

namespace cudf
{

__host__ __device__ inline string_view::string_view()
    : _data(""), _bytes(0), _length(0)
{}

__host__ __device__ inline string_view::string_view(const char* data, size_type bytes)
    : _data(data), _bytes(bytes), _length(UNKNOWN_STRING_LENGTH)
{}

//
__host__ __device__ inline size_type string_view::size_bytes() const
{
    return _bytes;
}

__device__ inline size_type string_view::length() const
{
    if( _length <= UNKNOWN_STRING_LENGTH )
        _length = strings::detail::characters_in_string(_data,_bytes);
    return _length;
}

__host__ __device__ inline const char* string_view::data() const
{
    return _data;
}

__host__ __device__ inline bool string_view::empty() const
{
    return _bytes == 0;
}

__host__ __device__ inline bool string_view::is_null() const
{
    return _data == nullptr;
}

// the custom iterator knows about UTF8 encoding
__device__ inline string_view::const_iterator::const_iterator(const string_view& str, size_type pos)
    : cpos{pos}, p{str.data()}, offset{str.byte_offset(pos)}
{}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator++()
{
    offset += bytes_in_utf8_byte((BYTE)p[offset]);
    ++cpos;
    return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator++(int)
{
    string_view::const_iterator tmp(*this);
    operator++();
    return tmp;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator+(string_view::const_iterator::difference_type off)
{
    const_iterator tmp(*this);
    while(off-- > 0)
        ++tmp;
    return tmp;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator+=(string_view::const_iterator::difference_type off)
{
    while(off-- > 0)
        operator++();
    return *this;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator--()
{
    while( bytes_in_utf8_byte((BYTE)p[--offset])==0 );
    --cpos;
    return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator--(int)
{
    string_view::const_iterator tmp(*this);
    operator--();
    return tmp;
}

__device__ inline string_view::const_iterator& string_view::const_iterator::operator-=(string_view::const_iterator::difference_type offset)
{
    while(offset-- > 0)
        operator--();
    return *this;
}

__device__ inline string_view::const_iterator string_view::const_iterator::operator-(string_view::const_iterator::difference_type offset)
{
    string_view::const_iterator tmp(*this);
    while(offset-- > 0)
        --tmp;
    return tmp;
}

__device__ inline bool string_view::const_iterator::operator==(const string_view::const_iterator& rhs) const
{
    return (p == rhs.p) && (cpos == rhs.cpos);
}

__device__ inline bool string_view::const_iterator::operator!=(const string_view::const_iterator& rhs) const
{
    return (p != rhs.p) || (cpos != rhs.cpos);
}

__device__ inline bool string_view::const_iterator::operator<(const string_view::const_iterator& rhs) const
{
    return (p == rhs.p) && (cpos < rhs.cpos);
}

// unsigned int can hold 1-4 bytes for the UTF8 char
__device__ inline char_utf8 string_view::const_iterator::operator*() const
{
    char_utf8 chr = 0;
    strings::detail::to_char_utf8(p + offset, chr);
    return chr;
}

__device__ inline size_type string_view::const_iterator::position() const
{
    return cpos;
}

__device__ inline size_type string_view::const_iterator::byte_offset() const
{
    return offset;
}

__device__ inline string_view::const_iterator string_view::begin() const
{
    return const_iterator(*this, 0);
}

__device__ inline string_view::const_iterator string_view::end() const
{
    return const_iterator(*this, length());
}

__device__ inline char_utf8 string_view::operator[](size_type pos) const
{
    unsigned int offset = byte_offset(pos);
    if(offset >= _bytes)
        return 0;
    char_utf8 chr = 0;
    strings::detail::to_char_utf8(data() + offset, chr);
    return chr;
}

__device__ inline size_type string_view::byte_offset(size_type pos) const
{
    size_type offset = 0;
    const char* sptr = _data;
    const char* eptr = sptr + _bytes;
    while( (pos > 0) && (sptr < eptr) )
    {
        size_type charbytes = bytes_in_utf8_byte((BYTE)*sptr++);
        if( charbytes )
            --pos;
        offset += charbytes;
    }
    return offset;
}

__device__ inline int string_view::compare(const string_view& in) const
{
    return compare(in.data(), in.size_bytes());
}

__device__ inline int string_view::compare(const char* data, size_type bytes) const
{
    const unsigned char* ptr1 = reinterpret_cast<const unsigned char*>(this->data());
    if(!ptr1)
        return -1;
    const unsigned char* ptr2 = reinterpret_cast<const unsigned char*>(data);
    if(!ptr2)
        return 1;
    size_type len1 = size_bytes();
    size_type idx = 0;
    for(; (idx < len1) && (idx < bytes); ++idx)
    {
        if(*ptr1 != *ptr2)
            return (int)*ptr1 - (int)*ptr2;
        ++ptr1;
        ++ptr2;
    }
    if(idx < len1)
        return 1;
    if(idx < bytes)
        return -1;
    return 0;
}

__device__ inline bool string_view::operator==(const string_view& rhs) const
{
    return compare(rhs) == 0;
}

__device__ inline bool string_view::operator!=(const string_view& rhs) const
{
    return compare(rhs) != 0;
}

__device__ inline bool string_view::operator<(const string_view& rhs) const
{
    return compare(rhs) < 0;
}

__device__ inline bool string_view::operator>(const string_view& rhs) const
{
    return compare(rhs) > 0;
}

__device__ inline bool string_view::operator<=(const string_view& rhs) const
{
    int rc = compare(rhs);
    return (rc == 0) || (rc < 0);
}

__device__ inline bool string_view::operator>=(const string_view& rhs) const
{
    int rc = compare(rhs);
    return (rc == 0) || (rc > 0);
}

__device__ inline size_type string_view::find(const string_view& str, size_type pos, int count) const
{
    return find(str.data(), str.size_bytes(), pos, count);
}

__device__ inline size_type string_view::find(const char* str, size_type bytes, size_type pos, int count) const
{
    const char* sptr = data();
    if(!str || !bytes)
        return -1;
    size_type nchars = length();
    if(count < 0)
        count = nchars;
    size_type end = pos + count;
    if(end < 0 || end > nchars)
       end = nchars;
    size_type spos = byte_offset(pos);
    size_type epos = byte_offset(end);

    size_type len2 = bytes;
    size_type len1 = (epos - spos) - len2 + 1;

    const char* ptr1 = sptr + spos;
    const char* ptr2 = str;
    for(size_type idx=0; idx < len1; ++idx)
    {
        bool match = true;
        for( size_type jdx=0; match && (jdx < len2); ++jdx )
            match = (ptr1[jdx] == ptr2[jdx]);
        if( match )
            return character_offset(idx+spos);
        ptr1++;
    }
    return -1;
}

__device__ inline size_type string_view::find(char_utf8 chr, size_type pos, int count) const
{
    char str[sizeof(char_utf8)];
    size_type chwidth = strings::detail::from_char_utf8(chr,str);
    return find(str,chwidth,pos,count);
}

__device__ inline size_type string_view::rfind(const string_view& str, size_type pos, int count) const
{
    return rfind(str.data(), str.size_bytes(), pos, count);
}

__device__ inline size_type string_view::rfind(const char* str, size_type bytes, size_type pos, int count) const
{
    const char* sptr = data();
    if(!str || !bytes)
        return -1;
    size_type sz = size_bytes();
    size_type nchars = length();
    size_type end = pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    size_type spos = byte_offset(pos);
    size_type epos = byte_offset(end);

    size_type len2 = bytes;
    size_type len1 = (epos - spos) - len2 + 1;

    const char* ptr1 = sptr + epos - len2;
    const char* ptr2 = str;
    for(int idx=0; idx < len1; ++idx)
    {
        bool match = true;
        for(size_type jdx=0; match && (jdx < len2); ++jdx)
            match = (ptr1[jdx] == ptr2[jdx]);
        if(match)
            return character_offset(epos - len2 - idx);
        ptr1--; // go backwards
    }
    return -1;
}

__device__ inline size_type string_view::rfind(char_utf8 chr, size_type pos, int count) const
{
    char str[sizeof(char_utf8)];
    size_type chwidth = strings::detail::from_char_utf8(chr,str);
    return rfind(str,chwidth,pos,count);
}

// parameters are character position values
__device__ inline string_view string_view::substr(size_type pos, size_type length) const
{
    size_type spos = byte_offset(pos);
    size_type epos = byte_offset(pos + length);
    if( epos > size_bytes() )
        epos = size_bytes();
    if(spos >= epos)
        return string_view("",0);
    return string_view(data()+spos,epos-spos);
}


__device__ inline size_type string_view::character_offset(size_type bytepos) const
{
    return strings::detail::characters_in_string(data(), bytepos);
}

namespace strings
{
namespace detail
{
__host__ __device__ inline size_type bytes_in_char_utf8(char_utf8 chr)
{
    size_type count = 1;
    count += (int)((chr & (unsigned)0x0000FF00) > 0);
    count += (int)((chr & (unsigned)0x00FF0000) > 0);
    count += (int)((chr & (unsigned)0xFF000000) > 0);
    return count;
}

__host__ __device__ inline size_type to_char_utf8(const char* pSrc, char_utf8 &chr)
{
    size_type chwidth = bytes_in_utf8_byte((BYTE)*pSrc);
    chr = (char_utf8)(*pSrc++) & 0xFF;
    if(chwidth > 1)
    {
        chr = chr << 8;
        chr |= ((char_utf8)(*pSrc++) & 0xFF); // << 8;
        if(chwidth > 2)
        {
            chr = chr << 8;
            chr |= ((char_utf8)(*pSrc++) & 0xFF); // << 16;
            if(chwidth > 3)
            {
                chr = chr << 8;
                chr |= ((char_utf8)(*pSrc++) & 0xFF); // << 24;
            }
        }
    }
    return chwidth;
}

__host__ __device__ inline size_type from_char_utf8(char_utf8 chr, char* dst)
{
    size_type chwidth = bytes_in_char_utf8(chr);
    for(size_type idx = 0; idx < chwidth; ++idx)
    {
        dst[chwidth - idx - 1] = (char)chr & 0xFF;
        chr = chr >> 8;
    }
    return chwidth;
}

// counts the number of characters in the given char array
__host__ __device__ inline size_type characters_in_string(const char* str, size_type bytes)
{
    if( (str==0) || (bytes==0) )
        return 0;
    //
    unsigned int nchars = 0;
    for(size_type idx = 0; idx < bytes; ++idx)
        nchars += (unsigned int)(((BYTE)str[idx] & 0xC0) != 0x80);
    return (size_type)nchars;
}

} // namespace detail
} // namespace strings
} // namespace cudf
