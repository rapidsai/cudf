/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef __ORC_KERNEL_PRIVATE_COMMON_H__
#define __ORC_KERNEL_PRIVATE_COMMON_H__

#include <stdio.h>
#include "kernel_orc.cuh"

#undef NOOP
#define NOOP (void)0;

#ifdef _DEBUG
#define DP0(...) if( threadIdx.x == 0) printf(__VA_ARGS__);
//#define DP(...)  {printf("id[%d]:", threadIdx.x); printf(__VA_ARGS__);}
#define DP(_fmt_, ...)  {printf("id[%d]:" _fmt_, threadIdx.x, __VA_ARGS__);}
#define DP0_INT(val) DP0(#val " :%d \n", val);
#else
#define DP0(...) NOOP
#define DP(...) NOOP
#define DP0_INT(...) NOOP
#endif

#define PRINT0(...) if( threadIdx.x == 0) printf(__VA_ARGS__);

template <typename T>
using EnableSigned = typename std::enable_if< std::is_signed<T>::value >::type*;
template <typename T>
using EnableUnSigned = typename std::enable_if< std::is_unsigned<T>::value >::type*;

#endif // __ORC_KERNEL_PRIVATE_COMMON_H__

