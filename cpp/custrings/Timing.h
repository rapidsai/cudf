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

#ifdef _WIN32

#define WINDOWS_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <windows.h> // QueryPerformanceFrequency, QueryPerformanceCounter

inline double GetTime()
{
	unsigned long long counter, frequency;
	QueryPerformanceCounter((LARGE_INTEGER*)(&counter));
	QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);

	return (double)counter / (double)frequency;
}

#else

#include <sys/time.h>
#include <unistd.h>

inline double GetTime()
{
	timeval tv;
	gettimeofday( &tv, NULL );
	return (double)(tv.tv_sec*1000000+tv.tv_usec)/1000000.0;
}

#endif


