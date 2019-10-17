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

#pragma once

#include <cuda_runtime.h>

// utf8 characters are 1-4 bytes
typedef unsigned int Char;

//
// This class represents and manages a single character array in device memory.
// The character array is expected as a UTF-8 encoded string.
// All index values and lengths are in characters and not bytes.
//
// All memory must be device memory provided by the caller and the class may not
// be created on the stack or new'd. Use the alloc_size() methods to determine how
// much memory is required for an instance to manage a string. Use the create_from()
// methods to create an instance over the provided memory segment. Caller must
// ensure the memory is large enough as per the alloc_size() value.
//
// Methods like replace() and strip() require new memory buffer to hold the
// resulting string. The insert() and append() methods require the string's
// original memory to be large enough hold the additional characters.
//
// The 'unsigned int' sizes here allow for a string to be 4 billion bytes long.
// This seems impractical given that the purpose of this class is to run parallel
// operations across many strings. You could only parallelize 8 strings of this
// size on a single 32GB GPU device.
//
class custring_view
{
    custring_view() {}; // prevent creating instance directly
                        // use the create_from methods to provide memory for the object to reside
    custring_view(const custring_view&) {};
    ~custring_view() {};
    custring_view& operator=(const custring_view&) { return *this; };

protected:
    unsigned int m_bytes;  // combining these two did not save space
    unsigned int m_chars;  // number of characters
    //char* m_data;        // all variable length data including char array;
                           // pointer is now calculated on demand

    __device__ inline static custring_view* create_from(void* buffer);
    __host__ __device__ inline void init_fields(unsigned int bytes);
    __device__ inline unsigned int offset_for_char_pos(unsigned int chpos) const;
    __device__ inline void offsets_for_char_pos(unsigned int& spos, unsigned int& epos) const;
    __device__ inline unsigned int char_offset(unsigned int bytepos) const;

public:

    // returns the amount of memory required to manage the given character array
    __host__ __device__ inline static unsigned int alloc_size(const char* data, unsigned int size);
    // returns the amount of memory needed to manage character array of this size
    __host__ __device__ inline static unsigned int alloc_size(unsigned int bytes, unsigned int chars);
    // these can be used to create instances in already allocated memory
    __host__ __device__ inline static custring_view* create_from(void* buffer, const char* data, unsigned int size);
    __device__ inline static custring_view* create_from(void* buffer, custring_view& str);
    // use this one create a single custring_view instance in device memory from a CPU character array
    __host__ inline static custring_view* create_from_host(void* devmem, const char* data, unsigned int size);

    // return how much memory is used by this instance
    __device__ inline unsigned int alloc_size() const;
    //
    __device__ inline unsigned int size() const;        // same as length()
    __device__ inline unsigned int length() const;      // number of bytes
    __device__ inline unsigned int chars_count() const; // number of characters
    __device__ inline char* data();            // raw pointer, use at your own risk
    __device__ inline const char* data() const;
    // returns true if string has no characters
    __device__ inline bool empty() const;

    // iterator is read-only
    class iterator
    {
        const char* p;
        unsigned int cpos, offset;
    public:
        __device__ inline iterator(custring_view& str,unsigned int initPos);
        __device__ inline iterator(const iterator& mit);
        __device__ inline iterator& operator++();
        __device__ inline iterator operator++(int);
        __device__ inline bool operator==(const iterator& rhs) const;
        __device__ inline bool operator!=(const iterator& rhs) const;
        __device__ inline Char operator*() const;
        __device__ inline unsigned int position() const;
        __device__ inline unsigned int byte_offset() const;
    };
    // iterator methods
    __device__ inline iterator begin();
    __device__ inline iterator end();

    // return character (UTF-8) at given position
    __device__ inline Char at(unsigned int pos) const;
    // this is read-only right now since modifying an individual character may change the memory requirements
    __device__ inline Char operator[](unsigned int pos) const;
    // return the byte offset for a character position
    __device__ inline unsigned int byte_offset_for(unsigned int pos) const;

    // return 0 if arg string matches
    // return <0 or >0 depending first different character
    __device__ inline int compare(const custring_view& str) const;
    __device__ inline int compare(const char* data, unsigned int bytes) const;

    __device__ inline bool operator==(const custring_view& rhs);
    __device__ inline bool operator!=(const custring_view& rhs);
    __device__ inline bool operator<(const custring_view& rhs);
    __device__ inline bool operator>(const custring_view& rhs);
    __device__ inline bool operator<=(const custring_view& rhs);
    __device__ inline bool operator>=(const custring_view& rhs);

    // return character position if arg string is contained in this string
    // return -1 if string is not found
    // (pos,pos+count) is the range of this string that is scanned
    __device__ inline int find( const custring_view& str, unsigned int pos=0, int count=-1 ) const;
    __device__ inline int find( const char* str, unsigned int bytes, unsigned int pos=0, int count=-1 ) const;
    __device__ inline int find( Char chr, unsigned int pos=0, int count=-1 ) const;
    // same as find() but searches from the end of this string
    __device__ inline int rfind( const custring_view& str, unsigned int pos=0, int count=-1 ) const;
    __device__ inline int rfind( const char* str, unsigned int bytes, unsigned int pos=0, int count=-1 ) const;
    __device__ inline int rfind( Char chr, unsigned int pos=0, int count=-1 ) const;
    // these are for parity with std::string
    __device__ inline int find_first_of( const custring_view& str, unsigned int pos=0 ) const;
    __device__ inline int find_first_of( const char* str, unsigned int bytes, unsigned int pos=0 ) const;
    __device__ inline int find_first_of( Char ch, unsigned int pos=0 ) const;
    __device__ inline int find_first_not_of( const custring_view& str, unsigned int pos=0 ) const;
    __device__ inline int find_first_not_of( const char* str, unsigned int bytes, unsigned int pos=0 ) const;
    __device__ inline int find_first_not_of( Char ch, unsigned int pos=0 ) const;
    __device__ inline int find_last_of( const custring_view& str, unsigned int pos=0 ) const;
    __device__ inline int find_last_of( const char* str, unsigned int bytes, unsigned int pos=0 ) const;
    __device__ inline int find_last_of( Char ch, unsigned int pos=0 ) const;
    __device__ inline int find_last_not_of( const custring_view& str, unsigned int pos=0 ) const;
    __device__ inline int find_last_not_of( const char* str, unsigned int bytes, unsigned int pos=0 ) const;
    __device__ inline int find_last_not_of( Char ch, unsigned int pos=0 ) const;

    // return substring based on character position and length
    // caller must provide memory for the resulting object
    __device__ inline custring_view* substr( unsigned int pos, unsigned int length, unsigned int step, void* mem );
    __device__ inline unsigned int substr_size( unsigned int pos, unsigned int length, unsigned int step=1 ) const;
    // copy the character array to the given device memory pointer
    __device__ inline unsigned int copy( char* str, int count, unsigned int pos=0 );

    // append string or character to this string
    // orginal string must have been created with enough memory for this operation
    __device__ inline custring_view& operator+=( const custring_view& str );
    __device__ inline custring_view& operator+=( Char chr );
    __device__ inline custring_view& operator+=( const char* str );
    // append argument string to this one
    __device__ inline custring_view& append( const char* str, unsigned int bytes );
    __device__ inline custring_view& append( const custring_view& str );
    __device__ inline custring_view& append( Char chr, unsigned int count=1 );
    __device__ inline unsigned int append_size( const char* str, unsigned int bytes ) const;
    __device__ inline unsigned int append_size( const custring_view& str ) const;
    __device__ inline unsigned int append_size( Char chr, unsigned int count=1 ) const;

    // insert the given string into the character position specified
    // orginal string must have been created with enough memory for this operation
    __device__ inline custring_view& insert( unsigned int pos, const char* data, unsigned int bytes );
    __device__ inline custring_view& insert( unsigned int pos, custring_view& str );
    __device__ inline custring_view& insert( unsigned int pos, unsigned int count, Char chr );
    __device__ inline unsigned int insert_size( const char* str, unsigned int bytes ) const;
    __device__ inline unsigned int insert_size( const custring_view& str ) const;
    __device__ inline unsigned int insert_size( Char chr, unsigned int count=1 ) const;

    // replace the given range of characters with the arg string
    // caller must provide memory for the resulting object
    __device__ inline custring_view* replace( unsigned int pos, unsigned int length, const char* data, unsigned int bytes, void* mem );
    __device__ inline custring_view* replace( unsigned int pos, unsigned int length, const custring_view& str, void* mem );
    __device__ inline custring_view* replace( unsigned int pos, unsigned int length, unsigned int count, Char chr, void* mem );
    __device__ inline unsigned int replace_size( unsigned int pos, unsigned int length, const char* data, unsigned int bytes ) const;
    __device__ inline unsigned int replace_size( unsigned int pos, unsigned int length, const custring_view& str ) const;
    __device__ inline unsigned int replace_size( unsigned int pos, unsigned int length, unsigned int count, Char chr ) const;

    // tokenizes string around the given delimiter string upto count
    // call with strs=0, will return the number of string tokens
    __device__ inline unsigned int split( const char* delim, unsigned int bytes, int count, custring_view** strs );
    __device__ inline unsigned int split_size( const char* delim, unsigned int bytes, int* sizes, int count ) const;
    __device__ inline unsigned int rsplit( const char* delim, unsigned int bytes, int count, custring_view** strs );
    __device__ inline unsigned int rsplit_size( const char* delim, unsigned int bytes, int* sizes, int count ) const;

    // return new string with given character from the beginning/end removed from this string
    // caller must provide memory for the resulting object
    __device__ inline custring_view* strip( const char* tgts, void* mem );
    __device__ inline unsigned int strip_size( const char* tgts ) const;
    __device__ inline custring_view* lstrip( const char* tgts, void* mem );
    __device__ inline unsigned int lstrip_size( const char* tgts ) const;
    __device__ inline custring_view* rstrip( const char* tgts, void* mem );
    __device__ inline unsigned int rstrip_size( const char* tgts ) const;

    // return numeric value represented by the characters in this string
    __device__ inline int stoi() const;
    __device__ inline long stol() const;
    __device__ inline unsigned long stoul() const;
    __device__ inline float stof() const;
    __device__ inline double stod() const;
    //
    __device__ inline static custring_view* ltos( long value, void* mem );
    __device__ inline static unsigned int ltos_size( long value );

    //
    __device__ inline bool starts_with( const char* str, unsigned int bytes ) const;
    __device__ inline bool starts_with( custring_view& str ) const;
    __device__ inline bool ends_with( const char* str, unsigned int bytes ) const;
    __device__ inline bool ends_with( custring_view& str ) const;

    // some utilities for handling individual UTF-8 characters
    __host__ __device__ inline static unsigned int bytes_in_char( Char chr );
    __host__ __device__ inline static unsigned int char_to_Char( const char* str, Char& chr );
    __host__ __device__ inline static unsigned int Char_to_char( Char chr, char* str );
    __host__ __device__ inline static unsigned int chars_in_string( const char* str, unsigned int bytes );
};

#include "custring_view.inl"
