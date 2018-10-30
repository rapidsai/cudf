#include <cstddef>
#include <vector>

//
// This maps directly to the methods in the cudastrings class in the cudastrings.py source file.
// It is a host object that manages vectors of strings stored on the device.
// Each operation performs (in parallel) against all strings in this instance.
//
class CUDAStrings
{
    class Impl;
    Impl* pImpl;

    // create placeholder for building return objects
    CUDAStrings(int count);

public:
    // Call this before using this class the first time in the process.
    // It sets up static elements like for the memory pool manager.
    static void initLibrary();

    // Create new strings instance.
    // Parameters are host strings that are copied into device memory.
    CUDAStrings(const char** strs, int count);
    // Parameter is a list of pointers and lengths to char-arrays in host or device memory.
    CUDAStrings(std::vector<std::pair<const char*,size_t> >& strs);
    CUDAStrings(std::pair<const char*,size_t>* strs, size_t count, bool devmem=true);

    // Destroys this instance and frees all of its strings
    ~CUDAStrings();

    // return the number of bytes used by this instance
    size_t memsize();
    // number of strings managed by this instance
    size_t size();
    
    // copy the list of strings back into the provided host memory
    int to_host(char** list, int start, int end);
    // create index for strings contained in this instance
    // pointers are to internal device memory
    int create_index(std::vector<std::pair<const char*,size_t> >& strs );

    // create a new instance containing only the strings at the specified positions
    CUDAStrings* sublist( int* pos, int count );
    // not yet implemented
    CUDAStrings* append( CUDAStrings* strs );
    CUDAStrings* append( std::pair<const char*,size_t>* strs, size_t count );
    // return a new instance without the specified strings
    CUDAStrings* remove_strings( int* pos, int count );
    // return a new instance based on this instance but without any null entries
    CUDAStrings* remove_nulls();

    // return the length of each string
    size_t len(int* lengths, bool todevice=true);

    // compare single arg string to all the strings
    size_t compare( const char* str, int* results, bool todevice=true );
    
    size_t stoi(int* results, bool todevice=true);   // returns integer values represented by each string
    size_t stof(float* results, bool todevice=true); // returns float values represented by each string

    // adds the given string(s) to this list of strings and returns as new strings
    CUDAStrings* cat( CUDAStrings* others, const char* separator, const char* narep=0);
    // concatenates all strings into one new string
    CUDAStrings* join( const char* delimiter, const char* narep=0 );
    
    // each string is split into a list of new strings
    int split( const char* delimiter, int maxsplit, std::vector<CUDAStrings*>& results);
    int rsplit( const char* delimiter, int maxsplit, std::vector<CUDAStrings*>& results);
    // each string is split into two strings on the first delimiter found
    // three strings are returned for each string: left-half, delimiter itself, right-half
    int partition( const char* delimiter, std::vector<CUDAStrings*>& results);
    int rpartition( const char* delimiter, std::vector<CUDAStrings*>& results);

    // split each string into a new column -- number of columns = string with the most delimiters
    int split_column( const char* delimiter, int maxsplit, std::vector<CUDAStrings*>& results);

    // return a specific character (as a string) by position for each string
    CUDAStrings* get(int pos);
    // repeat each string? Need to check what this actually supposed to do
    CUDAStrings* repeat(int count);
    // add padding to each string as specified by the parameters
    enum padside { left, right, both };
    CUDAStrings* pad(int width, padside side, const char* fillchar=0);
    CUDAStrings* ljust( int width, const char* fillchar=0 );
    CUDAStrings* center( int width, const char* fillchar=0 );
    CUDAStrings* rjust( int width, const char* fillchar=0 );
    // pads string with number with leading zeros
    CUDAStrings* zfill( int width );
    // this inserts new-line characters into each string
    CUDAStrings* wrap( int width );

    // returns a substring of each string
    CUDAStrings* slice( int start=0, int stop=-1, int step=1 );
    // inserts the specified string (repl) into each string
    CUDAStrings* slice_replace( const char* repl, int start=0, int stop=-1 );
    //
    CUDAStrings* replace( const char* str, const char* repl, int maxrepl=-1 );

    // remove specified character if found at the beginning of each string
    CUDAStrings* lstrip( const char* to_strip );
    // remove specified character if found at the beginning or end of each string
    CUDAStrings* strip( const char* to_strip );
    // remove specified character if found at the end each string
    CUDAStrings* rstrip( const char* to_strip );
    
    // return new strings with modified character case
    CUDAStrings* lower();
    CUDAStrings* upper();
    CUDAStrings* capitalize();
    CUDAStrings* swapcase();
    CUDAStrings* title();

    // search for a string within each string
    // the index/rindex method just use these too
    // return value is the number of positive (>=0) results
    size_t find( const char* str, int start, int end, int* results, bool todevice=true );
    size_t rfind( const char* str, int start, int end, int* results, bool todevice=true );
    size_t contains( const char* str, bool* results, bool todevice=true );

    //
    size_t hash(unsigned int* results, bool todevice=true);

    // translate characters in each string
    CUDAStrings* translate( std::pair<unsigned,unsigned>* table, size_t count );

    // sort by length and name sorts by length first
    enum sorttype { none=0, length=1, name=2 };
    // sorts the strings managed by this instance
    void sort( sorttype& st, bool ascending=true );

    // not implemented
    CUDAStrings* normalize( const char* form );

        // this does not work -- intention is to pass a device fn that would be called in parallel for each string
    typedef long(*transform_function)(size_t idx, const char* str, size_t bytes);
    int transform( transform_function& fn );
};
