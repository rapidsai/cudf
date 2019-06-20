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


#ifndef __ERROR_CUH
#define __ERROR_CUH

#include <cstdio>
#include <cstdlib>
#include <iostream>


#define CHECK_ERROR(rtv, expected_value, msg)                                            \
{                                                                                        \
    if (rtv != expected_value) {                                                         \
        fprintf(stderr, "ERROR on line %d of file %s: %s\n",  __LINE__, __FILE__, msg);  \
        std::cerr << "Error code " << rtv << std::endl;                                  \
        exit(1);                                                                         \
    }                                                                                    \
}

#endif // __ERROR_CUH
