/*
*/

#include <cstdlib>

namespace cudf
{

typedef unsigned char BYTE;

/**---------------------------------------------------------------------------*
 * @brief Returns the number of bytes used to represent the provided byte.
 * This could 0 to 4 bytes. 0 is returned for intermediate bytes within a
 * single character. For example, for the two-byte 0xC3A8 single character,
 * the first byte would return 2 and the second byte would return 0.
 *
 * @param byte Byte from an encoded character.
 * @return Number of bytes.
 *---------------------------------------------------------------------------**/
__host__ __device__ inline static size_type bytes_in_char_byte(BYTE byte)
{
    size_type count = 1;
    // no if-statements means no divergence
    count += (int)((byte & 0xF0) == 0xF0);
    count += (int)((byte & 0xE0) == 0xE0);
    count += (int)((byte & 0xC0) == 0xC0);
    count -= (int)((byte & 0xC0) == 0x80);
    return count;
}


/**---------------------------------------------------------------------------*
 * @brief Returns the number of bytes used in the provided char array by
 * searching for a null-terminator ('\0') byte.
 *
 * @param str Null-terminated array of chars.
 * @return Number of bytes.
 *---------------------------------------------------------------------------**/
__device__ inline static size_type string_length( const char* str )
{
    if( !str )
        return 0;
    size_type bytes = 0;
    while(*str++)
        ++bytes;
    return bytes;
}


__device__ inline string_view::string_view(const char* data, size_type bytes)
    : _data(data), _bytes(bytes)
{}

__device__ inline string_view::string_view(const char* data)
    : _data(data)
{
    _bytes = string_length(data);
}

//
__device__ inline size_type string_view::size() const
{
    return _bytes;
}

__device__ inline size_type string_view::length() const
{
    return _bytes;
}

__device__ inline size_type string_view::characters() const
{
    return chars_in_string(_data,_bytes);
}

__device__ inline const char* string_view::data() const
{
    return _data;
}

__device__ inline bool string_view::empty() const
{
    return _bytes == 0;
}

__device__ inline bool string_view::is_null() const
{
    return _data == nullptr;
}

// the custom iterator knows about UTF8 encoding
__device__ inline string_view::iterator::iterator(const string_view& str, size_type pos)
    : cpos(pos)
{
    p = str.data();
    offset = str.byte_offset_for(cpos);
}

__device__ inline string_view::iterator& string_view::iterator::operator++()
{
    offset += bytes_in_char_byte((BYTE)p[offset]);
    ++cpos;
    return *this;
}

// what is the int parm for?
__device__ inline string_view::iterator string_view::iterator::operator++(int)
{
    iterator tmp(*this);
    operator++();
    return tmp;
}

__device__ inline bool string_view::iterator::operator==(const string_view::iterator& rhs) const
{
    return (p == rhs.p) && (cpos == rhs.cpos);
}

__device__ inline bool string_view::iterator::operator!=(const string_view::iterator& rhs) const
{
    return (p != rhs.p) || (cpos != rhs.cpos);
}

// unsigned int can hold 1-4 bytes for the UTF8 char
__device__ inline Char string_view::iterator::operator*() const
{
    Char chr = 0;
    char_to_Char(p + offset, chr);
    return chr;
}

__device__ inline size_type string_view::iterator::position() const
{
    return cpos;
}

__device__ inline size_type string_view::iterator::byte_offset() const
{
    return offset;
}

__device__ inline string_view::iterator string_view::begin() const
{
    return iterator(*this, 0);
}

__device__ inline string_view::iterator string_view::end() const
{
    return iterator(*this, characters());
}

__device__ inline Char string_view::at(size_type pos) const
{
    unsigned int offset = byte_offset_for(pos);
    if(offset >= _bytes)
        return 0;
    Char chr = 0;
    char_to_Char(data() + offset, chr);
    return chr;
}

__device__ inline Char string_view::operator[](size_type pos) const
{
    return at(pos);
}

__device__ inline size_type string_view::byte_offset_for(size_type pos) const
{
    size_type offset = 0;
    const char* sptr = _data;
    const char* eptr = sptr + _bytes;
    while( (pos > 0) && (sptr < eptr) )
    {
        size_type charbytes = bytes_in_char_byte((BYTE)*sptr++);
        if( charbytes )
            --pos;
        offset += charbytes;
    }
    return offset;
}

__device__ inline int string_view::compare(const string_view& in) const
{
    return compare(in.data(), in.size());
}

__device__ inline int string_view::compare(const char* data, size_type bytes) const
{
    const unsigned char* ptr1 = reinterpret_cast<const unsigned char*>(this->data());
    if(!ptr1)
        return -1;
    const unsigned char* ptr2 = reinterpret_cast<const unsigned char*>(data);
    if(!ptr2)
        return 1;
    size_type len1 = size();
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
    return find(str.data(), str.size(), pos, count);
}

__device__ inline size_type string_view::find(const char* str, size_type bytes, size_type pos, int count) const
{
    const char* sptr = data();
    if(!str || !bytes)
        return -1;
    size_type nchars = characters();
    if(count < 0)
        count = nchars;
    size_type end = pos + count;
    if(end < 0 || end > nchars)
       end = nchars;
    size_type spos = byte_offset_for(pos);
    size_type epos = byte_offset_for(end);

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
            return char_offset(idx+spos);
        ptr1++;
    }
    return -1;
}

// maybe get rid of this one
__device__ inline size_type string_view::find(Char chr, size_type pos, int count) const
{
    size_type sz = size();
    size_type nchars = characters();
    if(count < 0)
        count = nchars;
    size_type end = pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    if(pos > end || chr == 0 || sz == 0)
        return -1;
    size_type spos = byte_offset_for(pos);
    size_type epos = byte_offset_for(end);
    //
    size_type chsz = bytes_in_char(chr);
    const char* sptr = data();
    const char* ptr = sptr + spos;
    size_type len = (epos - spos) - chsz;
    for(size_type idx = 0; idx <= len; ++idx)
    {
        Char ch = 0;
        char_to_Char(ptr++, ch);
        if(chr == ch)
            return chars_in_string(sptr, idx + spos);
    }
    return -1;
}

__device__ inline size_type string_view::rfind(const string_view& str, size_type pos, int count) const
{
    return rfind(str.data(), str.size(), pos, count);
}

__device__ inline size_type string_view::rfind(const char* str, size_type bytes, size_type pos, int count) const
{
    const char* sptr = data();
    if(!str || !bytes)
        return -1;
    size_type sz = size();
    size_type nchars = characters();
    size_type end = pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    size_type spos = byte_offset_for(pos);
    size_type epos = byte_offset_for(end);

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
            return char_offset(epos - len2 - idx);
        ptr1--; // go backwards
    }
    return -1;
}

__device__ inline size_type string_view::rfind(Char chr, size_type pos, int count) const
{
    size_type sz = size();
    size_type nchars = characters();
    if(count < 0)
        count = nchars;
    size_type end = pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    if(pos > end || chr == 0 || sz == 0)
        return -1;
    size_type spos = byte_offset_for(pos);
    size_type epos = byte_offset_for(end);

    size_type chsz = bytes_in_char(chr);
    const char* sptr = data();
    const char* ptr = sptr + epos - 1;
    size_type len = (epos - spos) - chsz;
    for(size_type idx = 0; idx < len; ++idx)
    {
        Char ch = 0;
        char_to_Char(ptr--, ch);
        if(chr == ch)
            return chars_in_string(sptr, epos - idx - 1);
    }
    return -1;
}


// parameters are character position values
__device__ inline string_view string_view::substr(size_type pos, size_type length) const
{
    size_type spos = byte_offset_for(pos);
    size_type epos = byte_offset_for(pos + length);
    if( epos > size() )
        epos = size();
    if(spos >= epos)
        return string_view("",0);
    length = epos - spos; // converts length to bytes
    return string_view(data()+spos,length);
}

__device__ inline size_type string_view::split(const char* delim, int count, string_view* strs) const
{
    const char* sptr = data();
    size_type sz = size();
    if(sz == 0)
    {
        if(strs && count)
            strs[0] = *this;
        return 1;
    }

    size_type bytes = string_length(delim);
    size_type delimCount = 0;
    size_type pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, pos + bytes);
    }

    size_type strsCount = delimCount + 1;
    size_type rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!strs)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    size_type dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    size_type nchars = characters();
    size_type spos = 0, sidx = 0;
    size_type epos = find(delim, bytes);
    while(epos >= 0)
    {
        if(sidx >= (count - 1)) // add this to the while clause
            break;
        strs[sidx++] = substr(spos, epos - spos);
        spos = epos + dchars;
        epos = find(delim, bytes, spos);
    }
    if((spos <= nchars) && (sidx < count))
        strs[sidx] = substr(spos, nchars - spos);
    //
    return rtn;
}


__device__ inline size_type string_view::rsplit(const char* delim, int count, string_view* strs) const
{
    const char* sptr = data();
    size_type sz = size();
    if(sz == 0)
    {
        if(strs && count)
            strs[0] = *this;
        return 1;
    }

    size_type bytes = string_length(delim);
    size_type delimCount = 0;
    size_type pos = find(delim, bytes);
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
    int epos = (int)characters(); // end pos is not inclusive
    int sidx = count - 1;          // index for strs array
    int spos = rfind(delim, bytes);
    while(spos >= 0)
    {
        if(sidx <= 0)
            break;
        //int spos = pos + (int)bytes;
        int len = epos - spos - dchars;
        strs[sidx--] = substr((unsigned int)spos+dchars, (unsigned int)len);
        epos = spos;
        spos = rfind(delim, bytes, 0, (unsigned int)epos);
    }
    if(epos >= 0)
        strs[0] = substr(0, epos);
    //
    return rtn;
}


__host__ __device__ inline size_type string_view::bytes_in_char(Char chr)
{
    size_type count = 1;
    count += (int)((chr & (unsigned)0x0000FF00) > 0);
    count += (int)((chr & (unsigned)0x00FF0000) > 0);
    count += (int)((chr & (unsigned)0xFF000000) > 0);
    return count;
}

__host__ __device__ inline size_type string_view::char_to_Char(const char* pSrc, Char &chr)
{
    size_type chwidth = bytes_in_char_byte((BYTE)*pSrc);
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

__host__ __device__ inline size_type string_view::Char_to_char(Char chr, char* dst)
{
    size_type chwidth = bytes_in_char(chr);
    for(size_type idx = 0; idx < chwidth; ++idx)
    {
        dst[chwidth - idx - 1] = (char)chr & 0xFF;
        chr = chr >> 8;
    }
    return chwidth;
}

// counts the number of characters in the given char array
__host__ __device__ inline size_type string_view::chars_in_string(const char* str, size_type bytes)
{
    if( (str==0) || (bytes==0) )
        return 0;
    //
    unsigned int nchars = 0;
    for(size_type idx = 0; idx < bytes; ++idx)
        nchars += (unsigned int)(((BYTE)str[idx] & 0xC0) != 0x80);
    return (size_type)nchars;
}

__device__ inline size_type string_view::char_offset(size_type bytepos) const
{
    return chars_in_string(data(), bytepos);
}

}