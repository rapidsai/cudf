/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory.h>
#include <cmath>

#include "./custring.cuh"

// 8-byte boundary
#define ALIGN_SIZE(v) (((v + 7) / 8) * 8)
// convenient byte type
typedef unsigned char BYTE;

// this allows overlapping memory segments
// useful for the methods that work within the memory they are provided
__host__ __device__ inline void _memmove(void* dest, void* source, size_t len)
{
    char* dst = (char*)dest;
    char* src = (char*)source;
    if(dst == src)
        return;
    if(dst < src)
    {
        while(len-- > 0)
            *dst++ = *src++;
        return;
    }
    dst = dst + len - 1;
    src = src + len - 1;
    while(len-- > 0)
        *dst-- = *src--;
}

// returns the number of bytes used to represent that char
__host__ __device__ inline unsigned int _bytes_in_char(BYTE byte)
{
    unsigned int count = 1;
    // no if-statements means no divergence
    count += (int)((byte & 0xF0) == 0xF0);
    count += (int)((byte & 0xE0) == 0xE0);
    count += (int)((byte & 0xC0) == 0xC0);
    count -= (int)((byte & 0xC0) == 0x80);
    return count;
}

// This puts the raw length values into byte array that has at least 2-bits
// per character. The 2-bits represent the char length in bytes:
//    00=1 byte, 01=2 bytes, 10=3 bytes, 11=4 bytes
// Example: string "Résidéntial" is 12 bytes but 10 characters
//          resolve this to lengths 1,2,0,1,1,1,2,0,1,1,1,1,1
//          and copy them into a 2-bit array.
__host__ __device__ inline void _fillCharsLengths(const char* str, unsigned int bytes, BYTE* charlens, unsigned int nchars)
{
    unsigned int blocks = (nchars + 3) / 4; // working with whole bytes
    unsigned int sidx = 0;
    for(unsigned int idx = 0; idx < blocks; ++idx)
    {
        unsigned int didx = idx * 4;
        BYTE len[4];
        // read the next four lengths
        for(int i = 0; i < 4; ++i)
        {
            unsigned int nchs = 0;
            while((nchs == 0) && (sidx < bytes)) // skip 'extra' chars (nchs=0)
                nchs = _bytes_in_char((BYTE)str[sidx++]);
            len[i] = 0;
            if(nchs && ((didx + i) < nchars))
                len[i] = nchs - 1; // encode 1:00b,2:01b,3:10b,4:11b
        }
        // convert to 2bit array
        BYTE val = len[0] << 6; // aa000000
        val += len[1] << 4;     // aabb0000
        val += len[2] << 2;     // aabbcc00
        val += len[3];          // aabbccdd
        charlens[idx] = val;
    }
}

// utility for methods allowing for multiple delimiters (e.g. strip)
__device__ inline bool is_one_of( const char* tgts, Char chr )
{
    Char tchr = 0;
    unsigned int cw = custring_view::char_to_Char(tgts,tchr);
    while( tchr )
    {
        if( tchr==chr )
            return true;
        tgts += cw;
        cw = custring_view::char_to_Char(tgts,tchr);
    }
    return false;
}

// vectorized-loader utility
class vloader
{
    char* _src;
    int _pos;
    unsigned long _ul;
public:
    __host__ __device__ inline vloader(char* src) : _src(src)
    {
        unsigned long pv = reinterpret_cast<unsigned long>(src);
        _pos = (int)(pv & 0x07);
        _src = src - _pos;
        _ul = reinterpret_cast<unsigned long*>(_src)[0];
    }
    __host__ __device__ inline unsigned char nextChar()
    {
        int offset = _pos & 0x07;
        if(!offset)
            _ul = reinterpret_cast<unsigned long*>(_src + _pos)[0];
        ++_pos;
        return reinterpret_cast<unsigned char*>(&_ul)[offset];
    }
};

// computes the size of the custring_view object needed to manage the given character array
__host__ __device__ inline unsigned int custring_view::alloc_size(const char* data, unsigned int bytes)
{
    unsigned int nsz = ALIGN_SIZE(sizeof(custring_view));
    //printf("as: custring_view(%lu)=%d\n",sizeof(custring_view),nsz);
    unsigned int sz = bytes + 1;
    unsigned int nchars = chars_in_string(data, bytes);
    if(nchars != bytes)
        sz += (nchars + 3) / 4;
    nsz += sz;
    //printf("as: +(%d/%d)=%d,%p\n",bytes,nchars,nsz,data);
    return nsz;
}

// shortcut method requires only the number of characters and the number of bytes
__host__ __device__ inline unsigned int custring_view::alloc_size(unsigned int bytes, unsigned int nchars)
{
    unsigned int nsz = ALIGN_SIZE(sizeof(custring_view));
    {
        unsigned int sz = bytes + 1;
        if(nchars != bytes)
            sz += (nchars + 3) / 4;
        nsz += sz;
    }
    return nsz;
}

// returns the number of bytes managed by this object
__device__ inline unsigned int custring_view::alloc_size() const
{
    unsigned int nsz = ALIGN_SIZE(sizeof(custring_view));
    {
        unsigned int bytes = size();
        unsigned int nchars = chars_count();
        unsigned int sz = bytes + 1;
        if(nchars != bytes)
            sz += (nchars + 3) / 4;
        nsz += sz;
    }
    return nsz;
}

// create a new instance within the memory provided
__host__ __device__ inline custring_view* custring_view::create_from(void* buffer, const char* data, unsigned int bytes)
{
    char* ptr = (char*)buffer;
    custring_view* dout = (custring_view*)ptr;
    ptr += ALIGN_SIZE(sizeof(custring_view));
    // buffer and data contents may overlap
    if(bytes > 0) // put data into proper spot first
        _memmove(ptr, (void*)data, bytes);
    // now set the rest of the fields
    dout->init_fields(bytes);
    //printf("cf: b=%p d=%p\n",buffer,ptr);
    return dout;
}

// create a new instance using the memory provided
__device__ inline custring_view* custring_view::create_from(void* buffer, custring_view& str)
{
    char* ptr = (char*)buffer;
    custring_view* dout = (custring_view*)ptr;
    ptr += ALIGN_SIZE(sizeof(custring_view));
    {
        unsigned int bytes = str.m_bytes;
        unsigned int chars = str.m_chars;
        //dout->m_data = ptr;
        dout->m_bytes = bytes;
        dout->m_chars = chars;
        if(bytes > 0)
        {
            //char* data1 = dout->m_data;
            //char* data2 = str.m_data;
            char* data1 = ((char*)dout) + ALIGN_SIZE(sizeof(custring_view));
            char* data2 = ((char*)&str) + ALIGN_SIZE(sizeof(custring_view));
            memcpy(data1, data2, bytes);
            data1[bytes] = 0;
            if(chars != bytes)
            {
                BYTE* charlens1 = (BYTE*)(data1 + bytes + 1);
                BYTE* charlens2 = (BYTE*)(data2 + bytes + 1);
                memcpy(charlens1, charlens2, (chars + 3) / 4);
            }
        }
    }
    return dout;
}

// create an (immutable) empty object
__device__ inline custring_view* custring_view::create_from(void* buffer)
{
    char* ptr = (char*)buffer;
    custring_view* dout = (custring_view*)ptr;
    ptr += ALIGN_SIZE(sizeof(custring_view));
    {
        //dout->m_data = ptr;
        dout->m_bytes = 0;
        dout->m_chars = 0;
    }
    return dout;
}

// create a single instance (device) from character array in host memory
__host__ inline custring_view* custring_view::create_from_host(void* devmem, const char* data, unsigned int size)
{
    if( data==0 || devmem==0 )
        return 0; // nulls is as nulls does Mrs Blue
    unsigned int alsz = alloc_size(data,size);
    char* buffer = new char[alsz];
    custring_view* hstr = create_from(buffer,data,size);
    cudaMemcpy(devmem,buffer,alsz,cudaMemcpyHostToDevice);
    delete buffer;
    return (custring_view*)devmem;
}

// called to finalize the metadata components
__host__ __device__ inline void custring_view::init_fields(unsigned int bytes)
{
    m_bytes = bytes;
    //char* data = m_data;
    char* data = ((char*)this) + ALIGN_SIZE(sizeof(custring_view));
    data[bytes] = 0; // null-terminate (for debug convenience)
    m_chars = chars_in_string(data, bytes);
    if(m_chars != bytes)
    {
        BYTE* charlens = (BYTE*)(data + bytes + 1);
        _fillCharsLengths(data, bytes, charlens, m_chars);
    }
}

__device__ inline unsigned int custring_view::offset_for_char_pos(unsigned int chpos) const
{
    if(chpos == 0)
        return 0;
    if(chpos >= m_chars)
        return m_bytes;
    if(m_chars == m_bytes)
        return chpos;
    //
    char* data = ((char*)this) + ALIGN_SIZE(sizeof(custring_view));
    BYTE* charlens = (BYTE*)(data + m_bytes + 1);
    int offset = 0;
    for(unsigned int idx=0; idx < chpos; ++idx)
    {
        BYTE lens = charlens[idx / 4];             // locate byte containing length
        int bits = (int)(idx & 3) * 2;             // locate bit pos for this character
        int len = (int)((lens >> (6 - bits)) & 3); // get length from bits
        offset += (len + 1);                       // decode 00b:1,01b:2,10b:3,11b:4
    }
    return offset;
}

// epos must be >= spos
__device__ inline void custring_view::offsets_for_char_pos(unsigned int& spos, unsigned int& epos) const
{
    if(epos > m_chars)
        epos = m_chars;
    if(m_chars == m_bytes)
        return;
    //
    char* data = ((char*)this) + ALIGN_SIZE(sizeof(custring_view));
    BYTE* charlens = (BYTE*)(data + m_bytes + 1);
    int offset = 0;
    for(unsigned int idx = 0; idx < spos; ++idx)
    {
        BYTE lens = charlens[idx / 4];
        int bits = (int)(idx & 3) * 2;
        int len = (int)((lens >> (6 - bits)) & 3);
        offset += (len + 1);
    }
    unsigned int tmp = offset;
    for(unsigned int idx = spos; idx < epos; ++idx)
    {
        BYTE lens = charlens[idx / 4];
        int bits = (int)(idx & 3) * 2;
        int len = (int)((lens >> (6 - bits)) & 3);
        offset += (len + 1);
    }
    spos = tmp;
    epos = offset;
}


//
__device__ inline unsigned int custring_view::size() const
{
    return m_bytes;
}

__device__ inline unsigned int custring_view::length() const
{
    return m_bytes;
}

__device__ inline unsigned int custring_view::chars_count() const
{
    return m_chars;
}

__device__ inline char* custring_view::data()
{
    //return m_data;
    return ((char*)this) + ALIGN_SIZE(sizeof(custring_view));
}

__device__ inline const char* custring_view::data() const
{
    //return m_data;
    return ((const char*)this) + ALIGN_SIZE(sizeof(custring_view));
}

__device__ inline bool custring_view::empty() const
{
    return m_chars == 0;
}

// the custom iterator knows about UTF8 encoding
__device__ inline custring_view::iterator::iterator(custring_view& str, unsigned int initPos)
    : p(0), cpos(0), offset(0)
{
    p = str.data();
    cpos = initPos;
    offset = str.offset_for_char_pos(cpos);
}

__device__ inline custring_view::iterator::iterator(const custring_view::iterator& mit)
    : p(mit.p), cpos(mit.cpos), offset(mit.offset)
{
}

__device__ inline custring_view::iterator& custring_view::iterator::operator++()
{
    offset += _bytes_in_char((BYTE)p[offset]);
    ++cpos;
    return *this;
}

// what is the int parm for?
__device__ inline custring_view::iterator custring_view::iterator::operator++(int)
{
    iterator tmp(*this);
    operator++();
    return tmp;
}

__device__ inline bool custring_view::iterator::operator==(const custring_view::iterator& rhs) const
{
    return (p == rhs.p) && (cpos == rhs.cpos);
}

__device__ inline bool custring_view::iterator::operator!=(const custring_view::iterator& rhs) const
{
    return (p != rhs.p) || (cpos != rhs.cpos);
}

// unsigned int can hold 1-4 bytes for the UTF8 char
__device__ inline Char custring_view::iterator::operator*() const
{
    Char chr = 0;
    char_to_Char(p + offset, chr);
    return chr;
}

__device__ inline unsigned int custring_view::iterator::position() const
{
    return cpos;
}

__device__ inline unsigned int custring_view::iterator::byte_offset() const
{
    return offset;
}

__device__ inline custring_view::iterator custring_view::begin()
{
    return iterator(*this, 0);
}

__device__ inline custring_view::iterator custring_view::end()
{
    return iterator(*this, chars_count());
}

__device__ inline Char custring_view::at(unsigned int pos) const
{
    unsigned int offset = offset_for_char_pos(pos);
    if(offset >= m_bytes)
        return 0;
    Char chr = 0;
    char_to_Char(data() + offset, chr);
    return chr;
}

__device__ inline Char custring_view::operator[](unsigned int pos) const
{
    return at(pos);
}

__device__ inline unsigned int custring_view::byte_offset_for(unsigned int pos) const
{
    return offset_for_char_pos(pos);
}

__device__ inline int custring_view::compare(const custring_view& in) const
{
    return compare(in.data(), in.size());
}

__device__ inline int custring_view::compare(const char* tgt, unsigned int bytes) const
{
    return custr::compare(data(),size(),tgt,bytes);
}

__device__ inline bool custring_view::operator==(const custring_view& rhs)
{
    return compare(rhs) == 0;
}

__device__ inline bool custring_view::operator!=(const custring_view& rhs)
{
    return compare(rhs) != 0;
}

__device__ inline bool custring_view::operator<(const custring_view& rhs)
{
    return compare(rhs) < 0;
}

__device__ inline bool custring_view::operator>(const custring_view& rhs)
{
    return compare(rhs) > 0;
}

__device__ inline bool custring_view::operator<=(const custring_view& rhs)
{
    int rc = compare(rhs);
    return (rc == 0) || (rc < 0);
}

__device__ inline bool custring_view::operator>=(const custring_view& rhs)
{
    int rc = compare(rhs);
    return (rc == 0) || (rc > 0);
}

__device__ inline int custring_view::find(const custring_view& str, unsigned int pos, int count) const
{
    return find(str.data(), str.size(), pos, count);
}

__device__ inline int custring_view::find(const char* str, unsigned int bytes, unsigned int pos, int count) const
{
    char* sptr = (char*)data();
    if(!str || !bytes)
        return -1;
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
       end = nchars;
    //if( pos > end )
    //    return -1;
    int spos = (int)offset_for_char_pos(pos);
    int epos = (int)offset_for_char_pos((unsigned int)end);

    int len2 = (int)bytes;
    int len1 = (epos - spos) - (int)len2 + 1;
    //if( len1 < 0 )
    //    return -1; // arg does not fit in search range

    char* ptr1 = sptr + spos;
    char* ptr2 = (char*)str;
    for(int idx=0; idx < len1; ++idx)
    {
        bool match = true;
        for( int jdx=0; match && (jdx < len2); ++jdx )
            match = (ptr1[jdx] == ptr2[jdx]);
        if( match )
            return (int)char_offset(idx+spos);
        ptr1++;
    }
    return -1;
}

// maybe get rid of this one
__device__ inline int custring_view::find(Char chr, unsigned int pos, int count) const
{
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    if(pos > end || chr == 0 || sz == 0)
        return -1;
    int spos = (int)offset_for_char_pos(pos);
    int epos = (int)offset_for_char_pos((unsigned int)end);
    //
    int chsz = (int)bytes_in_char(chr);
    char* sptr = (char*)data();
    char* ptr = sptr + spos;
    int len = (epos - spos) - chsz;
    for(int idx = 0; idx <= len; ++idx)
    {
        Char ch = 0;
        char_to_Char(ptr++, ch);
        if(chr == ch)
            return (int)chars_in_string(sptr, idx + spos);
    }
    return -1;
}

__device__ inline int custring_view::rfind(const custring_view& str, unsigned int pos, int count) const
{
    return rfind(str.data(), str.size(), pos, count);
}

__device__ inline int custring_view::rfind(const char* str, unsigned int bytes, unsigned int pos, int count) const
{
    char* sptr = (char*)data();
    if(!str || !bytes)
        return -1;
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    //if( pos > end )
    //    return -1; // bad parms
    int spos = (int)offset_for_char_pos(pos);
    int epos = (int)offset_for_char_pos(end);

    int len2 = (int)bytes;
    int len1 = (epos - spos) - len2 + 1;
    //if( len1 < 0 )
    //    return -1; // arg does not fit in search range

    char* ptr1 = sptr + epos - len2;
    char* ptr2 = (char*)str;
    for(int idx=0; idx < len1; ++idx)
    {
        bool match = true;
        for(int jdx=0; match && (jdx < len2); ++jdx)
            match = (ptr1[jdx] == ptr2[jdx]);
        if(match)
            return (int)char_offset(epos - len2 - idx);
        ptr1--; // go backwards
    }
    return -1;
}

__device__ inline int custring_view::rfind(Char chr, unsigned int pos, int count) const
{
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    if(pos > end || chr == 0 || sz == 0)
        return -1;
    int spos = (int)offset_for_char_pos(pos);
    int epos = (int)offset_for_char_pos(end);

    int chsz = (int)bytes_in_char(chr);
    char* sptr = (char*)data();
    char* ptr = sptr + epos - 1;
    int len = (epos - spos) - chsz;
    for(int idx = 0; idx < len; ++idx)
    {
        Char ch = 0;
        char_to_Char(ptr--, ch);
        if(chr == ch)
            return (int)chars_in_string(sptr, epos - idx - 1);
    }
    return -1;
}

__device__ inline int custring_view::find_first_of(const custring_view& str, unsigned int pos) const
{
    return find_first_of(str.data(), str.size(), pos);
}

__device__ inline int custring_view::find_first_of(const char* str, unsigned int bytes, unsigned int pos) const
{
    unsigned int sz = size();
    if(!sz || !bytes || !str)
        return -1;
    unsigned int nchars = chars_count();
    if(pos >= nchars)
        return -1;
    //
    for(unsigned int idx = pos; idx < nchars; ++idx)
    {
        Char ch1 = at(idx);
        bool match = false;
        for(unsigned int jdx = 0; jdx < bytes; ++jdx)
        {
            char* ptr = (char*)&str[jdx];
            Char ch2 = 0;
            unsigned int cw = char_to_Char(ptr, ch2);
            if(ch1 == ch2)
            {
                match = true;
                break;
            }
            jdx += cw - 1;
        }
        if(match)
            return (int)idx;
    }
    return -1;
}

__device__ inline int custring_view::find_first_of(Char ch, unsigned int pos) const
{
    return find(ch, pos);
}

__device__ inline int custring_view::find_first_not_of(const custring_view& str, unsigned int pos) const
{
    return find_first_not_of(str.data(), str.size(), pos);
}

__device__ inline int custring_view::find_first_not_of(const char* str, unsigned int bytes, unsigned int pos) const
{
    unsigned int sz = size();
    if(!sz || !bytes || !str)
        return -1;
    unsigned int nchars = chars_count();
    if(pos >= nchars)
        return -1;
    //
    for(unsigned int idx = pos; idx < nchars; ++idx)
    {
        Char ch1 = at(idx);
        bool match = false;
        for(unsigned int jdx = 0; jdx < bytes; ++jdx)
        {
            char* ptr = (char*)&str[jdx];
            Char ch2 = 0;
            unsigned int cw = char_to_Char(ptr, ch2);
            if(ch1 == ch2)
            {
                match = true;
                break;
            }
            jdx += cw - 1;
        }
        if(!match)
            return (int)idx;
    }
    return -1;
}

__device__ inline int custring_view::find_first_not_of(Char ch, unsigned int pos) const
{
    unsigned int sz = size();
    if(!sz)
        return -1;
    unsigned int nchars = chars_count();
    if(pos >= nchars)
        return -1;
    //
    for(unsigned int idx = pos; idx < nchars; ++idx)
    {
        if(ch != at(idx))
            return (int)idx;
    }
    return -1;
}

__device__ inline int custring_view::find_last_of(const custring_view& str, unsigned int pos) const
{
    return find_last_of(str.data(), str.size(), pos);
}

__device__ inline int custring_view::find_last_of(const char* str, unsigned int bytes, unsigned int pos) const
{
    unsigned int sz = size();
    if(!sz || !bytes || !str)
        return -1;
    unsigned int nchars = chars_count();
    if(pos >= nchars)
        return -1;
    //
    for(unsigned int idx = pos; idx < nchars; ++idx)
    {
        Char ch1 = at(nchars - idx - 1);
        bool match = false;
        for(unsigned int jdx = 0; jdx < bytes; ++jdx)
        {
            char* ptr = (char*)&str[jdx];
            Char ch2 = 0;
            unsigned int cw = char_to_Char(ptr, ch2);
            if(ch1 == ch2)
            {
                match = true;
                break;
            }
            jdx += cw - 1;
        }
        if(match)
            return (int)(nchars - idx - 1);
    }
    return -1;
}

__device__ inline int custring_view::find_last_of(Char ch, unsigned int pos) const
{
    return rfind(ch, pos);
}

__device__ inline int custring_view::find_last_not_of(const custring_view& str, unsigned int pos) const
{
    return find_last_not_of(str.data(), str.size(), pos);
}

__device__ inline int custring_view::find_last_not_of(const char* str, unsigned int bytes, unsigned int pos) const
{
    unsigned int sz = size();
    if(!sz || !bytes || !str)
        return -1;
    unsigned int nchars = chars_count();
    if(pos >= nchars)
        return -1;
    //
    for(int idx = (int)nchars - 1; idx >= pos; --idx)
    {
        Char ch1 = at(idx);
        bool match = false;
        for(unsigned int jdx = 0; jdx < bytes; ++jdx)
        {
            char* ptr = (char*)&str[jdx];
            Char ch2 = 0;
            unsigned int cw = char_to_Char(ptr, ch2);
            if(ch1 == ch2)
            {
                match = true;
                break;
            }
            jdx += cw - 1;
        }
        if(!match)
            return (int)idx;
    }
    return -1;
}

__device__ inline int custring_view::find_last_not_of(Char ch, unsigned int pos) const
{
    unsigned int sz = size();
    if(!sz)
        return -1;
    unsigned int nchars = chars_count();
    if(pos >= nchars)
        return -1;
    //
    for(int idx = (int)nchars - 1; idx >= pos; --idx)
    {
        if(ch != at(idx))
            return (int)idx;
    }
    return -1;
}

// parameters are character position values
__device__ inline custring_view* custring_view::substr(unsigned int pos, unsigned int length, unsigned int step, void* mem)
{
    custring_view* str = create_from(mem);
    unsigned int sz = size();
    unsigned int spos = offset_for_char_pos(pos);
    //if(spos > sz)
    //{
    //    str->init_fields(0);
    //    return str;
    //}
    unsigned int epos = offset_for_char_pos(pos + length);
    //if(epos > sz)
    //    epos = sz;
    if(spos >= epos)
    {
        str->init_fields(0);
        return str;
    }
    length = epos - spos; // converts length to bytes
    char* optr = str->data();
    char* sptr = data() + spos;
    if(step <= 1) // normal case
    {
        memcpy(optr, sptr, length);
        str->init_fields(length);
        return str;
    }
    // for step > 1 we need to capture specific characters from the string
    // copy individual chars to the new string
    unsigned int bytes = 0;
    for(unsigned int idx = 0; idx < length; idx += step)
    {
        Char chr = at(idx + pos);
        unsigned int chw = Char_to_char(chr, optr);
        bytes += chw;
        optr += chw;
    }
    str->init_fields(bytes);
    return str;
}

__device__ inline unsigned int custring_view::substr_size(unsigned int pos, unsigned int length, unsigned int step) const
{
    unsigned int sz = size();
    unsigned int spos = offset_for_char_pos(pos);
    //if(spos > sz)
    //    return alloc_size((unsigned)0, (unsigned)0);
    unsigned int epos = offset_for_char_pos(pos + length);
    //if(epos > sz)
    //    epos = sz;
    if( spos >= epos )
        return alloc_size((unsigned)0, (unsigned)0);
    length = epos - spos; // convert length to bytes
    char* sptr = (char*)data() + spos;
    if(step <= 1) // normal case
        return alloc_size(sptr, length);
    //
    unsigned int bytes = 0, nchars = 0;
    for(unsigned int idx = 0; idx < length; idx += step)
    {
        Char chr = at(idx + pos); // warning: check this may be slow, iterator may be faster
        ++nchars;
        bytes += bytes_in_char(chr);
    }
    return alloc_size(bytes, nchars);
}

__device__ inline unsigned int custring_view::copy(char* str, int count, unsigned int pos)
{
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    unsigned int end = pos + count;
    //if(end > nchars)
    //    end = nchars;
    //if(pos >= end)
    //    return 0;
    // convert positions to bytes
    unsigned int spos = offset_for_char_pos(pos);
    unsigned int epos = offset_for_char_pos(end);
    if( spos >= epos )
        return 0;
    // copy the data
    memcpy(str, data(), epos - spos);
    return end - pos;
}

// this is useful since operator+= takes only 1 argument
// inplace append; memory must be pre-allocated
__device__ inline custring_view& custring_view::append(const char* str, unsigned int bytes)
{
    unsigned int sz = size();
    char* sptr = data() + sz;
    memcpy(sptr, str, bytes);
    init_fields(sz + bytes);
    return *this;
}

__device__ inline custring_view& custring_view::append(Char chr, unsigned int count)
{
    unsigned int sz = size();
    char* sptr = data();
    unsigned int chsz = 0;
    for(unsigned idx = 0; idx < count; ++idx)
        chsz += Char_to_char(chr, sptr + sz + chsz);
    init_fields(sz + chsz);
    return *this;
}

__device__ inline custring_view& custring_view::append(const custring_view& in)
{
    return append(in.data(), in.size());
}

__device__ inline unsigned int custring_view::append_size(const char* str, unsigned int bytes) const
{
    unsigned int nbytes = size();
    unsigned int nchars = chars_count();
    if(bytes > 0)
    {
        nbytes += bytes;
        nchars += chars_in_string(str, bytes);
    }
    return alloc_size(nbytes, nchars);
}

__device__ inline unsigned int custring_view::append_size(const custring_view& in) const
{
    return append_size(in.data(), in.size());
}

__device__ inline unsigned int custring_view::append_size(Char chr, unsigned int count) const
{
    unsigned int bytes = size();
    unsigned int nchars = chars_count();
    if(count > 0)
    {
        nchars += count;
        bytes += count * bytes_in_char(chr);
    }
    return alloc_size(bytes, nchars);
}

__device__ inline custring_view& custring_view::operator+=(const custring_view& in)
{
    return append(in);
}

__device__ inline custring_view& custring_view::operator+=(Char chr)
{
    return append(chr);
}

// operators can only take one argument
__device__ inline custring_view& custring_view::operator+=(const char* str)
{
    char* sptr = (char*)str;
    unsigned int bytes = 0;
    while(*sptr++) // look for null-terminator
        ++bytes;
    return append(str, bytes);
}

__device__ inline custring_view& custring_view::insert(unsigned int pos, const char* str, unsigned int bytes)
{
    unsigned int sz = size();
    unsigned int spos = offset_for_char_pos(pos);
    if(spos > sz)
        return *this;
    char* sptr = this->data();
    if(sz == 0)
    { // insert whole string into this empty string
        if(bytes > 0)
            memcpy(sptr, str, bytes);
        init_fields(bytes);
        return *this;
    }

    unsigned int left = spos;
    unsigned int right = sz - spos;
    unsigned int nsz = left + right + bytes;
    // inplace insert; memory pre-allocated
    {
        char* optr = sptr + spos;
        if(right > 0) // move the right side first
            _memmove(optr + bytes, sptr + spos, right);
        if(bytes > 0)                 // fill in with
            memcpy(optr, str, bytes); // the new string
        init_fields(nsz);             // and re-init
    }
    //
    return *this;
}

__device__ inline custring_view& custring_view::insert(unsigned int pos, custring_view& in)
{
    return insert(pos, in.data(), in.size());
}

__device__ inline custring_view& custring_view::insert(unsigned int pos, unsigned int count, Char chr)
{
    unsigned int sz = size();
    unsigned int spos = offset_for_char_pos(pos);
    if(spos > sz || count == 0)
        return *this;
    char* sptr = this->data();
    unsigned int left = spos;
    unsigned int right = sz - spos;
    unsigned int bytes = bytes_in_char(chr) * count;
    unsigned int nsz = left + right + bytes;

    // inplace insert; memory pre-allocated
    {
        char* optr = sptr + spos;
        if(right > 0) // move the right side first
            _memmove(optr + bytes, sptr + spos, right);
        for(int idx = 0; idx < count; ++idx) // now fill in
            optr += Char_to_char(chr, optr);  // the new char
        init_fields(nsz);                     // and re-init
    }
    //
    return *this;
}

__device__ inline unsigned int custring_view::insert_size(const char* str, unsigned int bytes) const
{
    return append_size(str, bytes);
}

__device__ inline unsigned int custring_view::insert_size(const custring_view& in) const
{
    return append_size(in);
}

__device__ inline unsigned int custring_view::insert_size(Char chr, unsigned int count) const
{
    return append_size(chr, count);
}

// replace specified section with the given string
__device__ inline custring_view* custring_view::replace(unsigned int pos, unsigned int length, const char* data, unsigned int bytes, void* mem)
{
    // this will have upto 3 sections: left, arg-str, right
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    // handle all the boundaries
    unsigned int end = pos + length;
    if(end > nchars) // end is non-inclusize
        end = nchars;
    char* sptr = (char*)(this->data());
    custring_view* str = create_from(mem);
    char* optr = str->data();
    if((pos > end) ||  // range overlaps itself
       (pos > nchars)) // outside of string's range
    {
        memcpy(optr, sptr, sz);
        str->init_fields(sz);
        return str;
    }
    //
    unsigned int left = offset_for_char_pos(pos);
    unsigned int epos = offset_for_char_pos(end);
    unsigned int right = sz - epos;
    unsigned int nsz = left + bytes + right;

    memcpy(optr, sptr, left);
    optr += left;
    memcpy(optr, data, bytes);
    optr += bytes;
    memcpy(optr, sptr + epos, right);
    str->init_fields(nsz);
    //
    return str;
}

__device__ inline custring_view* custring_view::replace(unsigned int pos, unsigned int length, const custring_view& in, void* mem)
{
    return replace(pos, length, in.data(), in.size(), mem);
}

__device__ inline custring_view* custring_view::replace(unsigned int pos, unsigned int length, unsigned int count, Char chr, void* mem)
{
    // this will have upto 3 sections: left, arg-str, right
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    // handle all the boundaries
    unsigned int end = pos + length;
    if(end > nchars) // end is non-inclusize
        end = nchars;
    char* sptr = this->data();
    custring_view* str = create_from(mem);
    char* optr = str->data();
    if((pos > end) ||  // range overlaps itself
       (pos > nchars)) // outside of string's range
    {
        memcpy(optr, sptr, sz);
        str->init_fields(sz);
        return str;
    }
    //
    unsigned int chw = bytes_in_char(chr);
    unsigned int left = offset_for_char_pos(pos);
    unsigned int epos = offset_for_char_pos(end);
    unsigned int right = sz - epos;
    unsigned int nsz = left + (chw * count) + right;

    memcpy(optr, sptr, left);
    optr += left;
    for(unsigned int idx = 0; idx < count; ++idx)
        optr += Char_to_char(chr, optr);
    memcpy(optr, sptr + epos, right);
    str->init_fields(nsz);
    //
    return str;
}

//
__device__ inline unsigned int custring_view::replace_size(unsigned int pos, unsigned int length, const custring_view& in) const
{
    return replace_size(pos, length, in.data(), in.size());
}

//
__device__ inline unsigned int custring_view::replace_size(unsigned int pos, unsigned int length, const char* data, unsigned int bytes) const
{
    // this needs to over estimate
    // - if the new size is greater then we are ok
    // - if the new size is smaller we will not be able to replace in-place
    unsigned int sz = size();
    unsigned int chars = chars_count();
    // handle all the boundaries
    unsigned int end = pos + length;
    if(end > chars) // end is non-inclusize
        end = chars;
    if((pos > end) ||  // range overlaps itself
       (pos > chars))  // outside of string's range
        return alloc_size();
    //
    unsigned int left = offset_for_char_pos(pos);
    unsigned int epos = offset_for_char_pos(end);
    unsigned int right = sz - epos;

    unsigned int nbytes = left + bytes + right;
    unsigned int nchars = pos + chars_in_string(data, bytes) + (chars - end);
    return alloc_size(nbytes, nchars);
}

__device__ inline unsigned int custring_view::replace_size(unsigned int pos, unsigned int length, unsigned int count, Char chr) const
{
    unsigned int sz = size();
    unsigned int chars = chars_count();
    // handle all the boundaries
    unsigned int end = pos + length;
    if(end > chars) // end is non-inclusize
        end = chars;
    if((pos > end) ||  // range overlaps itself
       (pos > chars))  // outside of string's range
        return alloc_size();
    //
    unsigned int left = offset_for_char_pos(pos);
    unsigned int epos = offset_for_char_pos(end);
    unsigned int right = sz - epos;

    unsigned int nbytes = left + (bytes_in_char(chr) * count) + right;
    unsigned int nchars = pos + count + (chars - end);
    return alloc_size(nbytes, nchars);
}

__device__ inline unsigned int custring_view::split(const char* delim, unsigned int bytes, int count, custring_view** strs)
{
    char* sptr = data();
    unsigned int sz = size();
    if(sz == 0)
    {
        if(strs && count)
        {
            custring_view* str = create_from((void*)strs[0]);
            str->init_fields(0);
            strs[0] = str; // should be a noop
        }
        return 1;
    }

    unsigned int delimCount = 0;
    int pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, (unsigned int)pos + bytes);
    }

    unsigned int strsCount = delimCount + 1;
    unsigned int rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!strs)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    unsigned int dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    unsigned int nchars = chars_count();
    unsigned int spos = 0, sidx = 0;
    int epos = find(delim, bytes);
    while(epos >= 0)
    {
        if(sidx >= (count - 1)) // add this to the while clause
            break;
        int len = (unsigned int)epos - spos;
        void* str = (void*)strs[sidx++];
        substr(spos, len, 1, str);
        spos = epos + dchars;
        epos = find(delim, bytes, spos);
    }
    if((spos <= nchars) && (sidx < count))
        substr(spos, nchars - spos, 1, (void*)strs[sidx]);
    //
    return rtn;
}

//
__device__ inline unsigned int custring_view::split_size(const char* delim, unsigned int bytes, int *sizes, int count) const
{
    char* sptr = (char*)data();
    unsigned int sz = size();
    if(sz == 0)
    {
        if(sizes && count)
        {
            unsigned int ssz = alloc_size();
            sizes[0] = ssz;
            return ALIGN_SIZE(ssz);
        }
        return 1;
    }

    unsigned int delimCount = 0;
    int pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, (unsigned int)pos + bytes);
    }
    unsigned int strsCount = delimCount + 1;
    unsigned int rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!sizes)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    unsigned int dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    unsigned int nchars = chars_count();
    unsigned int total = 0;
    unsigned int spos = 0, sidx = 0;
    int epos = find(delim, bytes);
    while(epos >= 0)
    {
        if(sidx >= (count - 1)) // all but the last; this can be added to the while clause
            break;
        int len = (unsigned int)epos - spos;
        unsigned int ssz = substr_size(spos, len);
        sizes[sidx++] = ssz;
        total += ALIGN_SIZE(ssz);
        spos = epos + dchars;
        epos = find(delim, bytes, spos);
    }
    // handle the last string
    if((spos <= nchars) && (sidx < count))
    {
        unsigned int ssz = substr_size(spos, nchars - spos);
        sizes[sidx] = ssz;
        total += ALIGN_SIZE(ssz);
    }
    return total;
}

__device__ inline unsigned int custring_view::rsplit(const char* delim, unsigned int bytes, int count, custring_view** strs)
{
    char* sptr = data();
    unsigned int sz = size();
    if(sz == 0)
    {
        if(strs && count)
        {
            custring_view* str = create_from((void*)strs[0]);
            str->init_fields(0);
            strs[0] = str;
        }
        return 1;
    }

    unsigned int delimCount = 0;
    int pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, (unsigned int)pos + bytes);
    }

    unsigned int strsCount = delimCount + 1;
    unsigned int rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!strs)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    unsigned int dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    int epos = (int)chars_count(); // end pos is not inclusive
    int sidx = count - 1;          // index for strs array
    int spos = rfind(delim, bytes);
    while(spos >= 0)
    {
        if(sidx <= 0)
            break;
        //int spos = pos + (int)bytes;
        int len = epos - spos - dchars;
        void* str = (void*)strs[sidx--];
        substr((unsigned int)spos+dchars, (unsigned int)len, 1, str);
        epos = spos;
        spos = rfind(delim, bytes, 0, (unsigned int)epos);
    }
    if(epos >= 0)
    {
        void* str = (void*)strs[0];
        substr(0, epos, 1, str);
    }
    //
    return rtn;
}

__device__ inline unsigned int custring_view::rsplit_size(const char* delim, unsigned int bytes, int *sizes, int count) const
{
    char* sptr = (char*)data();
    unsigned int sz = size();
    if(sz == 0)
    {
        if(sizes && count)
        {
            unsigned int ssz = alloc_size();
            sizes[0] = ssz;
            return ALIGN_SIZE(ssz);
        }
        return 1;
    }

    unsigned int delimCount = 0;
    int pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, (unsigned int)pos + bytes);
    }

    unsigned int strsCount = delimCount + 1;
    unsigned int rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!sizes)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    unsigned int dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    unsigned int total = 0;        // total size of potential memory array
    int epos = (int)chars_count(); // end pos is not inclusive
    int sidx = count - 1;          // index for sizes array
    int spos = rfind(delim, bytes);
    while(spos >= 0)
    {
        if(sidx <= 0)
            break;
        //int spos = pos + (int)bytes;
        int len = epos - spos - dchars;
        unsigned int ssz = substr_size((unsigned int)spos+dchars, (unsigned int)len);
        sizes[sidx--] = ssz;
        total += ALIGN_SIZE(ssz);
        epos = spos;
        spos = rfind(delim, bytes, 0, (unsigned int)epos);
    }
    if(epos >= 0)
    {
        unsigned int ssz = substr_size(0, epos);
        sizes[0] = ssz;
        total += ALIGN_SIZE(ssz);
    }
    //
    return total;
}

__device__ inline custring_view* custring_view::strip(const char* tgts, void* mem)
{
    custring_view* str = create_from(mem);
    unsigned int sz = size();
    char* sptr = data();
    if(tgts == 0)
        tgts = " \n\t";
    char* optr = str->data();
    unsigned int nchars = chars_count();

    // count the leading bytes
    unsigned int lcount = 0;
    char* ptr = sptr;
    for(unsigned int idx = 0; idx < nchars; ++idx)
    {
        Char ch = 0;
        unsigned int cw = char_to_Char(ptr, ch);
        if( !is_one_of(tgts,ch) )
            break;
        ptr += cw;
        lcount += cw;
    }
    if(lcount == sz) // whole string was stripped
    {
        str->init_fields(0);
        return str;
    }
    // count trailing bytes
    unsigned int rcount = 0;
    ptr = sptr + sz; // point to the end
    for(unsigned int idx = 0; idx < nchars; ++idx)
    {
        while(_bytes_in_char((BYTE)*(--ptr)) == 0)
            ; // skip over 'extra' bytes
        Char ch = 0;
        unsigned int cw = char_to_Char(ptr, ch);
        if( !is_one_of(tgts,ch) )
            break;
        rcount += cw;
    }
    // left and right regions should not overlap
    if(lcount + rcount > sz)
    {
        str->init_fields(0);
        return str;
    }
    //
    unsigned int nsz = sz - rcount - lcount;
    memcpy(optr, sptr + lcount, nsz);
    str->init_fields(nsz);
    return str;
}

__device__ inline unsigned int custring_view::strip_size(const char* tgts) const
{
    unsigned int sz = size();
    char* sptr = (char*)data();
    if(tgts == 0)
        tgts = " \n\t";
    unsigned int nchars = chars_count();

    // count the leading chars
    unsigned int lcount = 0;
    char* ptr = sptr;
    for(unsigned int idx = 0; idx < nchars; ++idx)
    {
        Char ch = 0;
        unsigned int cw = char_to_Char(ptr, ch);
        if(!is_one_of(tgts,ch) )
            break;
        ptr += cw;
        lcount += cw;
    }
    if(lcount == sz) // whole string was stripped
        return alloc_size((unsigned)0, (unsigned)0);

    // count trailing characters
    unsigned int rcount = 0;
    ptr = sptr + sz; // point to the end
    for(unsigned int idx = 0; idx < nchars; ++idx)
    {
        while(_bytes_in_char((BYTE)*(--ptr)) == 0)
            ; // skip over 'extra' bytes
        Char ch = 0;
        unsigned int cw = char_to_Char(ptr, ch);
        if(!is_one_of(tgts,ch) )
            break;
        rcount += cw;
    }
    // left and right regions should not overlap
    if(lcount + rcount > sz)
        return alloc_size((unsigned)0, (unsigned)0); // not sure this can happen

    unsigned int span = sz - rcount - lcount;
    unsigned int nsz = alloc_size(sptr + lcount, span);
    return nsz;
}

__device__ inline custring_view* custring_view::lstrip(const char* tgts, void* mem)
{
    custring_view* str = create_from(mem);
    unsigned int sz = size();
    if(sz == 0)
    {
        str->init_fields(0);
        return str;
    }
    if(tgts == 0)
        tgts = " \n\t";
    unsigned int nchars = chars_count();
    // count the leading chars
    char* sptr = data();
    unsigned int count = 0;
    char* ptr = sptr;
    for(unsigned int idx = 0; idx < nchars; ++idx)
    {
        Char ch = 0;
        unsigned int cw = char_to_Char(ptr, ch);
        if( !is_one_of(tgts,ch) )
            break;
        ptr += cw;
        count += cw;
    }
    char* optr = str->data();
    unsigned int nsz = sz - count;
    memcpy(optr, sptr + count, nsz);
    str->init_fields(nsz);
    return str;
}

__device__ inline unsigned int custring_view::lstrip_size(const char* tgts) const
{
    unsigned int sz = size();
    if(sz == 0)
        return alloc_size();
    if(tgts == 0)
        tgts = " \n\t";
    unsigned int nchars = chars_count();
    char* sptr = (char*)data();
    char* ptr = sptr;
    unsigned int count = 0;
    for(unsigned int idx = 0; idx < nchars; ++idx)
    {
        Char ch = 0;
        unsigned int cw = char_to_Char(ptr, ch);
        if( !is_one_of(tgts,ch) )
            break;
        ptr += cw;
        count += cw;
    }
    return alloc_size(sptr + count, sz - count);
}

__device__ inline custring_view* custring_view::rstrip(const char* tgts, void* mem)
{
    custring_view* str = create_from(mem);
    unsigned int sz = size();
    char* sptr = data();
    if(tgts == 0)
        tgts = " \n\t";
    char* optr = str->data();
    unsigned int nchars = chars_count();
    char* ptr = sptr + sz;
    unsigned int count = 0;
    for(unsigned int idx = 0; idx < nchars; ++idx)
    {
        while(_bytes_in_char((BYTE)*(--ptr)) == 0)
            ; // skip over 'extra' bytes
        Char ch = 0;
        unsigned int cw = char_to_Char(ptr, ch);
        if( !is_one_of(tgts,ch) )
            break;
        count += cw;
    }
    unsigned int nsz = sz - count;
    memcpy(optr, sptr, nsz);
    str->init_fields(nsz);
    return str;
}

__device__ inline unsigned int custring_view::rstrip_size(const char* tgts) const
{
    unsigned int sz = size();
    char* sptr = (char*)data();
    if(tgts == 0)
        tgts = " \n\t";
    unsigned int nchars = chars_count();
    char* ptr = sptr + sz;
    unsigned int count = 0; // count from the end
    for(unsigned int idx = 0; idx < nchars; ++idx)
    {
        while(_bytes_in_char((BYTE)*(--ptr)) == 0)
            ; // skip over 'extra' bytes
        Char ch = 0;
        unsigned int cw = char_to_Char(ptr, ch);
        if( !is_one_of(tgts,ch) )
            break;
        count += cw;
    }
    return alloc_size(sptr, sz - count);
}

// these expect only numbers (ascii charset)
__device__ inline int custring_view::stoi() const
{
    return custr::stoi(data(),size());
}

__device__ inline long custring_view::stol() const
{
    return custr::stol(data(),size());
}

__device__ inline unsigned long custring_view::stoul() const
{
    return custr::stoul(data(),size());
}

__device__ inline float custring_view::stof() const
{
    return custr::stof(data(),size());
}

__device__ inline double custring_view::stod() const
{
    return custr::stod(data(),size());
}

__device__ inline custring_view* custring_view::ltos( long value, void* mem )
{
    if( value==0 )
        return create_from(mem,"0",1);
    char* str = (char*)mem;
    char* ptr = str;
    bool sign = value < 0;
    if( sign )
        value = -value;
    while( value > 0 )
    {
        char ch = '0' + (value % 10);
        *ptr++ = ch;
        value = value/10;
    }
    if( sign )
        *ptr++ = '-';
    // number is backwards, so let's reverse it
    int len = (int)(ptr-str);
    for( int j=0; j<(len/2); ++j )
    {
        char ch1 = str[j];
        char ch2 = str[len-j-1];
        str[j] = ch2;
        str[len-j-1] = ch1;
    }
    return create_from(str,str,len);
}

__device__ inline unsigned int custring_view::ltos_size( long value )
{
    if( value==0 )
        return alloc_size(1,1);
    bool sign = value < 0;
    if( sign )
        value = -value;
    int digits = 0; // count the digits
    while( value > 0 )
    {
        ++digits;
        value = value/10;
    }
    int bytes = digits + (int)sign;
    return alloc_size(bytes,bytes);
}

//
__device__ inline bool custring_view::starts_with(const char* str, unsigned int bytes) const
{
    if(bytes > size())
        return false;
    char* sptr = (char*)data();
    for(unsigned int idx = 0; idx < bytes; ++idx)
    {
        if(*sptr++ != *str++)
            return false;
    }
    return true;
}

__device__ inline bool custring_view::starts_with(custring_view& in) const
{
    return starts_with(in.data(), in.size());
}

__device__ inline bool custring_view::ends_with(const char* str, unsigned int bytes) const
{
    unsigned int sz = size();
    if(bytes > sz)
        return false;
    char* sptr = (char*)data() + sz - bytes; // point to the end
    for(unsigned int idx = 0; idx < bytes; ++idx)
    {
        if(*sptr++ != *str++)
            return false;
    }
    return true;
}

__device__ inline bool custring_view::ends_with(custring_view& str) const
{
    unsigned int sz = size();
    unsigned int bytes = str.size();
    if(bytes > sz)
        return false;
    return find(str, sz - bytes) >= 0;
}

__host__ __device__ inline unsigned int custring_view::bytes_in_char(Char chr)
{
    unsigned int count = 1;
    // no if-statements means no divergence
    count += (int)((chr & (unsigned)0x0000FF00) > 0);
    count += (int)((chr & (unsigned)0x00FF0000) > 0);
    count += (int)((chr & (unsigned)0xFF000000) > 0);
    return count;
}

__host__ __device__ inline unsigned int custring_view::char_to_Char(const char* pSrc, Char &chr)
{
    unsigned int chwidth = _bytes_in_char((BYTE)*pSrc);
    chr = (Char)(*pSrc++) & 0xFF;
    if(chwidth > 1)
    {
        chr = chr << 8;
        chr |= ((Char)(*pSrc++) & 0xFF); // << 8;
        if(chwidth > 2)
        {
            chr = chr << 8;
            chr |= ((Char)(*pSrc++) & 0xFF); // << 16;
            if(chwidth > 3)
            {
                chr = chr << 8;
                chr |= ((Char)(*pSrc++) & 0xFF); // << 24;
            }
        }
    }
    return chwidth;
}

__host__ __device__ inline unsigned int custring_view::Char_to_char(Char chr, char* dst)
{
    unsigned int chwidth = bytes_in_char(chr);
    for(unsigned int idx = 0; idx < chwidth; ++idx)
    {
        dst[chwidth - idx - 1] = (char)chr & 0xFF;
        chr = chr >> 8;
    }
    return chwidth;
}

// counts the number of character in the first bytes of the given char array
__host__ __device__ inline unsigned int custring_view::chars_in_string(const char* str, unsigned int bytes)
{
    if( (str==0) || (bytes==0) )
        return 0;
    //
    unsigned int nchars = 0;
    for(unsigned int idx = 0; idx < bytes; ++idx)
        nchars += (unsigned int)(((BYTE)str[idx] & 0xC0) != 0x80);
    return nchars;
    //unsigned int align = (unsigned long)str & 3;
    //str = str - align;                                                // these are pre-byte-swapped:
    //unsigned int m1 = ((unsigned int)0xFFFFFFFF) << (align*8);        // 0xFFFFFFFF, 0xFFFFFF00, 0xFFFF0000, 0xFF000000
    //unsigned int m2 = (unsigned int)(0x0080808080L >> ((4-align)*8)); // 0x00000000, 0x00000080, 0x00008080, 0x00808080
    //bytes += align; // adjust for alignment
    //for( unsigned int idx=0; idx < bytes/4; idx++ ) // read 4 bytes at a time
    //{
    //    unsigned int vl = ((unsigned int*)(str))[idx];
    //    vl = (vl & m1) | m2; // mask for alignment
    //    nchars += (vl & (unsigned int)0xC0000000) != (unsigned int)0x80000000;
    //    nchars += (vl & (unsigned int)0x00C00000) != (unsigned int)0x00800000;
    //    nchars += (vl & (unsigned int)0x0000C000) != (unsigned int)0x00008000;
    //    nchars += (vl & (unsigned int)0x000000C0) != (unsigned int)0x00000080;
    //    m1 = (unsigned int)0xFFFFFFFF; // the rest are
    //    m2 = (unsigned int)0x00000000; // already aligned
    //}
    //// finish off the end
    //unsigned int end = (bytes/4)*4;
    //for( unsigned int idx=0; idx < (bytes & 3); ++idx )
    //{
    //    unsigned char uc = (unsigned char)str[end+idx];
    //    nchars += ((uc & 0xC0) != 0x80);
    //}
    //return nchars;
}

__device__ inline unsigned int custring_view::char_offset(unsigned int bytepos) const
{
    if(m_bytes == m_chars)
        return bytepos;
    return chars_in_string(data(), bytepos);
}