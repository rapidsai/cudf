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

#include <Python.h>

//
PyObject* ncat_createCategoryFromBuffer( PyObject* self, PyObject* args );
PyObject* ncat_destroyCategory( PyObject* self, PyObject* args );
PyObject* ncat_size( PyObject* self, PyObject* args );
PyObject* ncat_keys_size( PyObject* self, PyObject* args );
PyObject* ncat_get_keys( PyObject* self, PyObject* args );
PyObject* ncat_keys_cpointer( PyObject* self, PyObject* args );
PyObject* ncat_keys_type( PyObject* self, PyObject* args );
PyObject* ncat_get_values( PyObject* self, PyObject* args );
PyObject* ncat_values_cpointer( PyObject* self, PyObject* args );
PyObject* ncat_get_indexes_for_key( PyObject* self, PyObject* args );
PyObject* ncat_to_type( PyObject* self, PyObject* args );
PyObject* ncat_gather_type( PyObject* self, PyObject* args );
PyObject* ncat_merge_category( PyObject* self, PyObject* args );
PyObject* ncat_add_keys( PyObject* self, PyObject* args );
PyObject* ncat_remove_keys( PyObject* self, PyObject* args );
PyObject* ncat_remove_unused( PyObject* self, PyObject* args );
PyObject* ncat_set_keys( PyObject* self, PyObject* args );
PyObject* ncat_gather( PyObject* self, PyObject* args );
PyObject* ncat_gather_values( PyObject* self, PyObject* args );
PyObject* ncat_gather_and_remap( PyObject* self, PyObject* args );
PyObject* UumLL( PyObject* self, PyObject* args );

