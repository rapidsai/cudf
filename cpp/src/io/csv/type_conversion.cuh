/* Copyright 2017 NVIDIA Corporation.  All rights reserved. */

#ifndef CONVERSION_FUNCTIONS_CUH
#define CONVERSION_FUNCTIONS_CUH

#include <cuda_runtime_api.h>



//---------------------------------------------------------------------------
//				Helper functions
//---------------------------------------------------------------------------

__host__ __device__
bool isDigit(char data) {
	if ( data < '0' ) return false;
	if ( data > '9' ) return false;

	return true;
}


__host__ __device__
void adjustForWhitespaceAndQuotes(const char *data, long* start_idx, long* end_idx, char quotechar='\0') {
  while ((*start_idx < *end_idx) && (data[*start_idx] == ' ' || data[*start_idx] == quotechar)) {
    (*start_idx)++;
  }
  while ((*start_idx < *end_idx) && (data[*end_idx] == ' ' || data[*end_idx] == quotechar)) {
    (*end_idx)--;
  }
}

template<typename T>
__host__ __device__
bool isBooleanValue(T value, int32_t* boolValues, int32_t count) {
	for (int i = 0; i < count; ++i) {
		if (value == static_cast<T>(boolValues[i])) {
			return true;
		}
	}
	return false;
}

//---------------------------------------------------------------------------



template<typename T>
__host__ __device__
T convertStrtoInt(const char *data, long start_idx, long end_idx, char thousands='\0') {

	T answer = (T)0;

	// if the start and end indexs are the same, then it is a single digit value
	if (start_idx == end_idx) {
		answer = (data[start_idx] -'0');
		return answer;
	}

	bool negative=false;
	if(data[start_idx]=='-'){
		negative=true;
		start_idx++;
	}

	// the data is in little ending, so the last item of data is the lowest digit
	int powSize = 0;
	long idx = end_idx;

	while(idx > (start_idx - 1))
	{
		if (data[idx] != thousands) {
			answer += (data[idx] -'0') * pow(10, powSize);
			++powSize;
		}

		--idx;
	}

	if (negative==true)
		answer *=-1;

    return answer;
}


template<typename T>
__host__ __device__
T convertStrtoFloat(char *data, long start_idx, long end_idx, char decimal, char thousands='\0') {

	T answer = (T)0.0;
	// removePrePostWhiteSpaces(data, &start_idx, &end_idx);


	// check for single digit conversions
	if (start_idx == end_idx) {
		answer = (data[start_idx] -'0');
		return answer;
	}

	// trim leading and trailing spaces
	if (data[start_idx] == ' ')
		++start_idx;

	if (data[end_idx] == ' ')
		--end_idx;

	bool negative=false;
	if(data[start_idx]=='-'){
		negative=true;
		start_idx++;
	}

	// find the decimal point - might not be one
	long decimal_pt = end_idx;
	long d_idx = start_idx;
	int found = 0;

	while ( (d_idx < (end_idx +1)) && ! found  ) {
		if ( data[d_idx] == decimal) {
			decimal_pt = d_idx;
			found = 1;
		}
		++d_idx;
	}

	// work on upper part
	long idx = decimal_pt;
	int powSize = 0;

	if ( idx >= start_idx ) {
		if (data[idx] == decimal)
			--idx;

		while(idx > (start_idx - 1))
		{
			if (data[idx] != thousands) {
				answer += (data[idx] -'0') * pow(10, powSize);
				++powSize;
			}
			--idx;
		}
	}

	//lower part - work left to right
	if ( found ) {
		powSize = -1;
		idx = decimal_pt +1;

		while(idx < (end_idx + 1))
		{
			if (data[idx] != thousands) {
				answer += (data[idx] -'0') * pow(10, powSize);
				--powSize;
			}

			++idx;
		}
	}

	if (negative==true)
		answer *=-1;


    return answer;
}


/**
 * Convert a date (MM/YYYY or DD/MM/YYYY) into a date64
 */
/*
__host__ __device__
int64_t convertStrtoDate(char *data, long start_idx, long end_idx) {

	// removePrePostWhiteSpaces(data, &start_idx, &end_idx);

	static unsigned short days[12] = {0,  31,  60,  91, 121, 152, 182, 213, 244, 274, 305, 335};

	int day 	= 01;
	int month 	= 01;
	int year 	= 1970;

	//  determine format  MM/YYYY  or DD/MM/YYYY by looking at size
	if (end_idx - start_idx < 8) {

		//find the "/"
		long slash_idx = start_idx;

		while (data[slash_idx] !='/' && slash_idx < end_idx)
			++slash_idx;

		month = convertStrtoInt<int>(data, start_idx, (slash_idx - 1));
		year  = convertStrtoInt<int>(data, (slash_idx + 1), end_idx);

	} else {

		//find the "/"
		long slash_idx = start_idx;

		while (data[slash_idx] !='/' && slash_idx < end_idx)
			++slash_idx;

		long slash_2_idx = slash_idx + 1;

		while (data[slash_2_idx] !='/' && slash_2_idx < end_idx)
			++slash_2_idx;

		day 	= convertStrtoInt<int>(data, start_idx, 		(slash_idx  - 1));
		month  	= convertStrtoInt<int>(data, (slash_idx + 1), 	(slash_2_idx - 1));
		year  	= convertStrtoInt<int>(data, (slash_2_idx + 1), end_idx);
	}

	// years since epoch
	int ye = year - 1970;

	// 1972 was a leap year, so how many between date and 1972?
	int lpy = (year - 1972)/4 + 1;

	// compute days since epoch
	int64_t days_e = ((ye - lpy) * 365) + (lpy * 366);

	// is this a leap year?
	if ( year % 4 == 0 && month > 2)
		days_e++;

	// months since January
	int me = month - 01;

	// days up to start of month
	days_e += days[me];

	// now just add days, but not full days since this one is not over
	days_e += (day -1);

	// scoot back to midnight
	--days_e;

	// now convert to seconds
	int64_t t = (days_e * 24) * 60 * 60;

	return t;
}
*/

__forceinline__
__host__ __device__ uint32_t rotl32( uint32_t x, int8_t r )
{
  return (x << r) | (x >> (32 - r));
}

__forceinline__
__host__ __device__ uint32_t fmix32( uint32_t h )
{
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

//MurmurHash3_32 implementation from https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.//MurmurHash3_32 implementation from https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
//
// This is a modified version of what is used for hash-join.  The change is at accept
// a char * key and range (start and end) so that the large raw CSV data pointer
// could be used
__host__ __device__
int32_t convertStrtoHash(const char * key, long start_idx, long end_idx, uint32_t m_seed)
{

    int len = (end_idx - start_idx) + 1;				// +1 since it is an inclusive range

    const uint8_t * const data = (const uint8_t*)key;
    int nblocks = len / 4;
    uint32_t h1 = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    //----------
    // body
    const uint32_t * const blocks = (const uint32_t *)(data + nblocks*4);
    for(int i = -nblocks; i; i++)
    {
        uint32_t k1 = blocks[i];//getblock32(blocks,i);
        k1 *= c1;
        k1 = rotl32(k1,15);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl32(h1,13);
        h1 = h1*5+0xe6546b64;
    }
    //----------
    // tail
    int processed_len = nblocks * 4;
    int left		= len - processed_len;
    long pad_idx	= end_idx - left + 1;
	char tail[4];

	for ( int idx = 0; idx < 4; idx ++) {
		if ( pad_idx + idx > end_idx)
			tail[idx] = 0;
		else
			tail[idx] = data[pad_idx + idx];
	}

    //const uint8_t * tail = (const uint8_t*)(data + nblocks*4);
    uint32_t k1 = 0;
    switch(len & 3)
    {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = rotl32(k1,15); k1 *= c2; h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
}





#endif
