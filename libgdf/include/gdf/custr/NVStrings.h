/*
* Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#include <cstddef>
#include <vector>

class NVStringsImpl;
//
// This maps indirectly to the methods in the nvstrings class in the nvstrings.py source file.
// It is a host object that manages vectors of strings stored in device memory.
// Each operation performs (in parallel) against all strings in this instance.
//
class NVStrings
{
    NVStringsImpl* pImpl;

    // internal helper ctor
    NVStrings(unsigned int count);

public:
    
    // Create new strings instance.
    // Parameters are host strings that are copied into device memory.
    NVStrings(const char** strs, unsigned int count);
    // Parameter is a list of pointers and lengths to char-arrays in host or device memory.
    NVStrings(std::vector<std::pair<const char*,size_t> >& strs);
    NVStrings(std::pair<const char*,size_t>* strs, unsigned int count, bool devmem=true);

    // Destroys this instance and frees all of its strings
    ~NVStrings();

    // return the number of device bytes used by this instance
    size_t memsize();
    // number of strings managed by this instance
    unsigned int size();
    
    // copy the list of strings back into the provided host memory
    int to_host(char** list, int start, int end);
    // create index for device strings contained in this instance
    // array must hold at least size() elements
    int create_index(std::pair<const char*,size_t>* strs, bool devmem=true );
    // create bit-array identifying the null strings
    int create_null_bitarray( unsigned char* bitarray, bool emptyIsNull=false, bool todevice=true );

    // create a new instance containing only the strings at the specified positions
    NVStrings* sublist( unsigned int* pos, unsigned int count );
    // return a new instance without the specified strings
    NVStrings* remove_strings( unsigned int* pos, unsigned int count );
    
    // return the length of each string
    unsigned int len(int* lengths, bool todevice=true);

    // compare single arg string to all the strings
    unsigned int compare( const char* str, int* results, bool todevice=true );
    
    unsigned int stoi(int* results, bool todevice=true);   // returns integer values represented by each string
    unsigned int stof(float* results, bool todevice=true); // returns float values represented by each string

    // adds the given string(s) to this list of strings and returns as new strings
    NVStrings* cat( NVStrings* others, const char* separator, const char* narep=0);
    // concatenates all strings into one new string
    NVStrings* join( const char* delimiter, const char* narep=0 );
    
    // each string is split into a list of new strings
    int split( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    int rsplit( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);
    // each string is split into two strings on the first delimiter found
    // three strings are returned for each string: left-half, delimiter itself, right-half
    int partition( const char* delimiter, std::vector<NVStrings*>& results);
    int rpartition( const char* delimiter, std::vector<NVStrings*>& results);

    // split each string into a new column -- number of columns = string with the most delimiters
    int split_column( const char* delimiter, int maxsplit, std::vector<NVStrings*>& results);

    // return a specific character (as a string) by position for each string
    NVStrings* get(unsigned int pos);
    // repeat each string? Need to check what this actually supposed to do
    NVStrings* repeat(unsigned int count);
    // add padding to each string as specified by the parameters
    enum padside { left, right, both };
    NVStrings* pad(unsigned int width, padside side, const char* fillchar=0);
    NVStrings* ljust( unsigned int width, const char* fillchar=0 );
    NVStrings* center( unsigned int width, const char* fillchar=0 );
    NVStrings* rjust( unsigned int width, const char* fillchar=0 );
    // pads string with number with leading zeros
    NVStrings* zfill( unsigned int width );
    // this inserts new-line characters into each string
    NVStrings* wrap( unsigned int width );

    // returns a substring of each string
    NVStrings* slice( int start=0, int stop=-1, int step=1 );
    // inserts the specified string (repl) into each string
    NVStrings* slice_replace( const char* repl, int start=0, int stop=-1 );
    //
    NVStrings* replace( const char* str, const char* repl, int maxrepl=-1 );

    // remove specified character if found at the beginning of each string
    NVStrings* lstrip( const char* to_strip );
    // remove specified character if found at the beginning or end of each string
    NVStrings* strip( const char* to_strip );
    // remove specified character if found at the end each string
    NVStrings* rstrip( const char* to_strip );
    
    // return new strings with modified character case
    NVStrings* lower();
    NVStrings* upper();
    NVStrings* capitalize();
    NVStrings* swapcase();
    NVStrings* title();

    // search for a string within each string
    // the index/rindex method just use these too
    // return value is the number of positive (>=0) results
    unsigned int find( const char* str, int start, int end, int* results, bool todevice=true );
    unsigned int rfind( const char* str, int start, int end, int* results, bool todevice=true );
    unsigned int contains( const char* str, bool* results, bool todevice=true );

    //
    unsigned int hash(unsigned int* results, bool todevice=true);

    // translate characters in each string
    NVStrings* translate( std::pair<unsigned,unsigned>* table, unsigned int count );
    // not implemented
    NVStrings* normalize( const char* form );

    // sort by length and name sorts by length first
    enum sorttype { none=0, length=1, name=2 };
    // sorts the strings managed by this instance
    void sort( sorttype& st, bool ascending=true );

    // output strings to stdout
    void print( int pos=0, int end=-1, int maxwidth=-1, const char* delimiter = "\n" );
    // for performance analysis
    void printTimingRecords();
};
