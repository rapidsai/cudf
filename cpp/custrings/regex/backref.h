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
#include <string>
#include <regex>
#include <cstdlib>
#include <vector>
#include <thrust/pair.h>

//
// Find back-ref index and position values.
// Returns modified string without back-ref indicators.
// Example:
//    for input string:    'hello \2 and \1'
//    the returned pairs:  (2,6),(1,11)
//    returned string is:  'hello  and '
//
std::string parse_backrefs( const char* str, std::vector<thrust::pair<int,int> >& brefs )
{
    std::string s = str;
    std::smatch m;
    std::regex ex("(\\\\\\d+)"); // this searches for backslash-number(s)
    std::string rtn;             // result without refs
    int bytepos = 0;
    while( std::regex_search( s, m, ex ) )
    {
        if( m.size()==0 )
            break;
        std::pair<int,int> item;
        std::string bref = m[0];
        int pos = (int)m.position(0);
        int len = (int)bref.length();
        bytepos += pos;
        item.first = std::atoi(bref.c_str()+1); // back-ref index number
        item.second = bytepos;                  // position within the string
        rtn += s.substr(0,pos);
        s = s.substr(pos + len);
        brefs.push_back(item);
    }
    if( !s.empty() ) // add the remainder
        rtn += s;    // of the string
    return rtn;
}

