/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "base_category.h"
#include <cstddef>

typedef unsigned char BYTE;

// forward reference for private data
template<typename T> class numeric_category_impl;

//
template<typename T>
class numeric_category : base_category_type
{
    numeric_category_impl<T>* pImpl;

    numeric_category();
    numeric_category( const numeric_category& );

public:

    numeric_category( const T* items, size_t count, const BYTE* nulls=nullptr );
    ~numeric_category();

    numeric_category<T>* copy();

    size_t size();
    size_t keys_size();

    const T* keys();
    const int* values();
    const BYTE* nulls_bitmask();
    bool has_nulls();
    bool keys_have_null();

    void print(const char* prefix="", const char* delimiter=" ");
    const char* get_type_name();

    const T get_key_for(int idx);
    bool is_value_null(int idx);

    int get_index_for(T key);
    size_t get_indexes_for(T key, int* result);
    size_t get_indexes_for_null_key(int* result);

    numeric_category<T>* add_keys( const T* items, size_t count, const BYTE* nulls=nullptr );
    numeric_category<T>* remove_keys( const T* items, size_t count, const BYTE* nulls=nullptr );
    numeric_category<T>* remove_unused_keys();
    numeric_category<T>* set_keys( const T* items, size_t count, const BYTE* nulls=nullptr );
    numeric_category<T>* merge( numeric_category<T>& cat );

    numeric_category<T>* gather(const int* indexes, size_t count );
    numeric_category<T>* gather_and_remap(const int* indexes, size_t count );
    numeric_category<T>* gather_values(const int* indexes, size_t count );

    // results/nulls must be able to hold size() entries
    void to_type( T* results, BYTE* nulls=nullptr );
    void gather_type( const int* indexes, size_t count, T* results, BYTE* nulls=nullptr );
};
