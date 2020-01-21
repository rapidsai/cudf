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

#include <Python.h>
#include <vector>
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdexcept>
#include <limits>
#include <cuda_runtime.h>
#include <rmm/rmm.h>
#include <nvstrings/numeric_category.h>

//
class DataHandler
{
    PyObject* pyobj;
    std::string errortext;
    unsigned int type_width;
    std::string dtype_name;
    void* host_data;
    void* dev_data;
    unsigned int count;

public:
    // we could pass in and check the type too (optional parameter)
    DataHandler( PyObject* obj, const char* validate_type=nullptr )
    : pyobj(obj), count(0), type_width(0), host_data(nullptr), dev_data(nullptr)
    {
        if( pyobj == Py_None )
            return; // not an error (e.g. nulls bitmask)

        std::string name = pyobj->ob_type->tp_name;
        if( name.compare("DeviceNDArray")==0 )
        {
            PyObject* pyasize = PyObject_GetAttr(pyobj,PyUnicode_FromString("alloc_size"));
            PyObject* pysize = PyObject_GetAttr(pyobj,PyUnicode_FromString("size"));
            PyObject* pydtype = PyObject_GetAttr(pyobj,PyUnicode_FromString("dtype"));
            PyObject* pydcp = PyObject_GetAttr(pyobj,PyUnicode_FromString("device_ctypes_pointer"));
            pyobj = PyObject_GetAttr(pydcp,PyUnicode_FromString("value"));
            //printf("dnda: size=%d, alloc_size=%d\n",(int)PyLong_AsLong(pysize),(int)PyLong_AsLong(pyasize));
            count = (unsigned int)PyLong_AsLong(pysize);
            if( count > 0 )
                type_width = PyLong_AsLong(pyasize)/count;
            //printf("dnda: count=%d, twidth=%d\n",(int)count,(int)type_width);
            if( pyobj != Py_None )
            {
                dev_data = PyLong_AsVoidPtr(pyobj);
                dtype_name = PyUnicode_AsUTF8(PyObject_Str(pydtype));
            }
        }
        else if( name.compare("numpy.ndarray")==0 )
        {
            PyObject* pyasize = PyObject_GetAttr(pyobj,PyUnicode_FromString("nbytes"));
            PyObject* pysize = PyObject_GetAttr(pyobj,PyUnicode_FromString("size"));
            PyObject* pydtype = PyObject_GetAttr(pyobj,PyUnicode_FromString("dtype"));
            PyObject* pydcp = PyObject_GetAttr(pyobj,PyUnicode_FromString("ctypes"));
            pyobj = PyObject_GetAttr(pydcp,PyUnicode_FromString("data"));
            //printf("nda: size=%d, alloc_size=%d\n",(int)PyLong_AsLong(pysize),(int)PyLong_AsLong(pyasize));
            count = (unsigned int)PyLong_AsLong(pysize);
            if( count > 0 )
                type_width = PyLong_AsLong(pyasize)/count;
            //printf("nda: count=%d, twidth=%d\n",(int)count,(int)type_width);
            if( pyobj != Py_None )
            {
                host_data = PyLong_AsVoidPtr(pyobj);
                dtype_name = PyUnicode_AsUTF8(PyObject_Str(pydtype));
            }
        }
        else if( name.compare("int")==0 ) // devptr
        {
            dev_data = PyLong_AsVoidPtr(pyobj);
            dtype_name = validate_type;
            count = std::numeric_limits<unsigned int>::max();
        }
        else
        {
            errortext = "unknown_type: ";
            errortext += name;
        }
        if( errortext.empty() && validate_type &&
            (dtype_name.compare(0,std::strlen(validate_type),validate_type)!=0) )
        {
            errortext = "argument must be of type ";
            errortext += validate_type;
        }
    }

    //
    ~DataHandler()
    {
        if( dev_data && host_data )
            cudaFree(dev_data);
    }

    //
    bool is_error()               { return !errortext.empty(); }
    const char* get_error_text()  { return errortext.c_str(); }
    unsigned int get_count()      { return count; }
    unsigned int get_type_width() { return type_width; }
    const char* get_dtype_name()  { return dtype_name.c_str(); }
    bool is_device_type()         { return dev_data && !host_data; }

    void* get_values()
    {
        if( dev_data || !host_data )
            return dev_data;
        cudaMalloc(&dev_data, count * type_width);
        cudaMemcpy(dev_data, host_data, count * type_width, cudaMemcpyHostToDevice);
        return dev_data;
    }
    void results_to_host()
    {
        if( host_data && dev_data )
            cudaMemcpy(host_data, dev_data, count * type_width, cudaMemcpyDeviceToHost);
    }
};

// from cudf's cpp/src/utilities/type_dispatcher.hpp
template<class functor_t, typename... Ts>
constexpr decltype(auto) type_dispatcher( const char* stype, functor_t fn, Ts&&... args )
{
    std::string dtype = stype;
    if( dtype.compare("int32")==0 )
        return fn.template operator()<int>(std::forward<Ts>(args)...);
    if( dtype.compare("int64")==0 )
        return fn.template operator()<long>(std::forward<Ts>(args)...);
    if( dtype.compare("float32")==0 )
        return fn.template operator()<float>(std::forward<Ts>(args)...);
    if( dtype.compare("float64")==0 )
        return fn.template operator()<double>(std::forward<Ts>(args)...);
    if( dtype.compare("int8")==0 )
        return fn.template operator()<char>(std::forward<Ts>(args)...);
    if( dtype.compare(0,10,"datetime64")==0 )
        return fn.template operator()<long>(std::forward<Ts>(args)...);
    //
    std::string msg = "invalid dtype in nvcategory dispatcher: ";
    msg += stype;
    throw std::runtime_error(msg);
}

template<typename T>
T pyobj_convert(PyObject* pyobj) { return 0; }
template<> int pyobj_convert<int>(PyObject* pyobj) { return (int)PyLong_AsLong(pyobj); };
template<> long pyobj_convert<long>(PyObject* pyobj) { return (long)PyLong_AsLong(pyobj); };
template<> float pyobj_convert<float>(PyObject* pyobj) { return (float)PyFloat_AsDouble(pyobj); };
template<> double pyobj_convert<double>(PyObject* pyobj) { return (double)PyFloat_AsDouble(pyobj); };
template<> char pyobj_convert<char>(PyObject* pyobj) { return (char)PyLong_AsLong(pyobj); };

template<typename T>
PyObject* convert_pyobj(T) { return nullptr; }
template<> PyObject* convert_pyobj<int>(int v) { return PyLong_FromLong((long)v); };
template<> PyObject* convert_pyobj<long>(long v) { return PyLong_FromLong(v); };
template<> PyObject* convert_pyobj<float>(float v) { return PyFloat_FromDouble((double)v); };
template<> PyObject* convert_pyobj<double>(double v) { return PyFloat_FromDouble(v); };
template<> PyObject* convert_pyobj<char>(char v) { return PyLong_FromLong((long)v); };


struct create_functor
{
    void* data;
    size_t count;
    BYTE* nulls;
    template<typename T>
    void* operator()()
    {
        T* items = reinterpret_cast<T*>(data);
        auto result = new numeric_category<T>(items,count,nulls);
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

struct base_functor
{
    base_category_type* obj_ptr;
    base_functor(base_category_type* obj_ptr) : obj_ptr(obj_ptr) {}
};

//
PyObject* ncat_createCategoryFromBuffer( PyObject* self, PyObject* args )
{
    PyObject* pydata = PyTuple_GetItem(args,0);
    DataHandler data(pydata);
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"category.create: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    PyObject* pynulls = PyTuple_GetItem(args,1);
    DataHandler nulls(pynulls);
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( data.get_dtype_name(), create_functor{data.get_values(),data.get_count(),reinterpret_cast<BYTE*>(nulls.get_values())} );
    //printf("category ctor(%p)\n",result);
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr(result);
}

// called by destructor in python class
PyObject* ncat_destroyCategory( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    Py_BEGIN_ALLOW_THREADS
    //printf("category dtor(%p)\n",tptr);
    delete tptr;
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

struct size_functor : base_functor
{
    size_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> size_t operator()() { return (reinterpret_cast<numeric_category<T>*>(obj_ptr))->size(); }
};

PyObject* ncat_size( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    size_t count;
    Py_BEGIN_ALLOW_THREADS
    count = type_dispatcher( tptr->get_type_name(), size_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(count);
}

struct keys_size_functor : base_functor
{
    keys_size_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> size_t operator()() { return (reinterpret_cast<numeric_category<T>*>(obj_ptr))->keys_size(); }
};

PyObject* ncat_keys_size( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    size_t count;
    Py_BEGIN_ALLOW_THREADS
    count = type_dispatcher( tptr->get_type_name(), keys_size_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(count);
}

struct keys_cpointer_functor : base_functor
{
    keys_cpointer_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> const void* operator()() { return (reinterpret_cast<numeric_category<T>*>(obj_ptr))->keys(); }
};

PyObject* ncat_keys_cpointer( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    const void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), keys_cpointer_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr(const_cast<void*>(result));
}

struct values_cpointer_functor : base_functor
{
    values_cpointer_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> const int* operator()() { return (reinterpret_cast<numeric_category<T>*>(obj_ptr))->values(); }
};

PyObject* ncat_values_cpointer( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    const int* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), values_cpointer_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

struct nulls_cpointer_functor : base_functor
{
    nulls_cpointer_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> const BYTE* operator()() { return (reinterpret_cast<numeric_category<T>*>(obj_ptr))->nulls_bitmask(); }
};

PyObject* ncat_nulls_cpointer( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    const BYTE* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), nulls_cpointer_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

struct get_keys_functor : base_functor
{
    get_keys_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T>
    void operator()(void* keys)
    {
        numeric_category<T>* this_ptr = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        if( keys )
            cudaMemcpy(keys, this_ptr->keys(), this_ptr->keys_size()*sizeof(T), cudaMemcpyDeviceToDevice);
    }
};

struct keys_to_list_functor : base_functor
{
    keys_to_list_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T>
    PyObject* operator()()
    {
        numeric_category<T>* this_ptr = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        size_t count = this_ptr->keys_size();
        T* keys;
        Py_BEGIN_ALLOW_THREADS
        keys = new T[count];
        cudaMemcpy(keys, this_ptr->keys(), count*sizeof(T), cudaMemcpyDeviceToHost);
        Py_END_ALLOW_THREADS
        PyObject* list = PyList_New(count);
        for( size_t idx=0; idx < count; ++idx )
        {
            if( idx==0 && this_ptr->keys_have_null() )
            {
                Py_INCREF(Py_None);
                PyList_SetItem(list, idx, Py_None);
                continue;
            }
            PyObject* pykey = convert_pyobj<T>(keys[idx]);
            PyList_SetItem(list, idx, pykey);
        }
        delete keys;
        return list;
    }
};

PyObject* ncat_get_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    DataHandler data(PyTuple_GetItem(args,1),tptr->get_type_name());
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"get_keys: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->get_type_name(), keys_size_functor(tptr) );
    void* results = data.get_values();
    if( results )
    {
        if( count > data.get_count() )
        {
            PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld %s values", count, tptr->get_type_name());
            Py_RETURN_NONE;
        }
        Py_BEGIN_ALLOW_THREADS
        type_dispatcher( tptr->get_type_name(), get_keys_functor(tptr), results );
        if( !data.is_device_type() )
            data.results_to_host();
        Py_END_ALLOW_THREADS
        Py_RETURN_NONE;
    }
    // special-case when no output buffer is given: return a PyList of values
    // this helps with debugging and is used by the nvcategory._str_() method
    PyObject* list = type_dispatcher( tptr->get_type_name(), keys_to_list_functor(tptr) );
    return list;
}

PyObject* ncat_keys_type( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    std::string type_name;
    Py_BEGIN_ALLOW_THREADS
    type_name = tptr->get_type_name();
    Py_END_ALLOW_THREADS
    return PyUnicode_FromString(type_name.c_str());
}

struct get_values_functor : base_functor
{
    get_values_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T>
    void operator()(int* values)
    {
        numeric_category<T>* this_ptr = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        cudaMemcpy(values, this_ptr->values(), this_ptr->size()*sizeof(int), cudaMemcpyDeviceToDevice);
    }
};

PyObject* ncat_get_values( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    DataHandler data(PyTuple_GetItem(args,1),"int32");
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"get_values: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->get_type_name(), size_functor(tptr) );
    int* results = reinterpret_cast<int*>(data.get_values());
    if( results )
    {
        if( count > data.get_count() )
        {
            PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld int32 values", count);
            Py_RETURN_NONE;
        }
        Py_BEGIN_ALLOW_THREADS
        type_dispatcher( tptr->get_type_name(), get_values_functor(tptr), results );
        if( !data.is_device_type() )
            data.results_to_host();
        Py_END_ALLOW_THREADS
        Py_RETURN_NONE;
    }
    // all this extra code to support calling values() with no argument
    // and returning a python list of the values
    BYTE* nulls = nullptr;
    Py_BEGIN_ALLOW_THREADS
    const int* d_values = type_dispatcher( tptr->get_type_name(), values_cpointer_functor(tptr) );
    const BYTE* d_nulls = type_dispatcher( tptr->get_type_name(), nulls_cpointer_functor(tptr) );
    size_t byte_count = (count+7)/8;
    results = new int[count];
    cudaMemcpy(results, d_values, count*sizeof(int), cudaMemcpyDeviceToHost);
    if( d_nulls )
    {
        nulls = new BYTE[byte_count];
        cudaMemcpy(nulls, d_nulls, byte_count, cudaMemcpyDeviceToHost);
    }
    Py_END_ALLOW_THREADS
    PyObject* list = PyList_New(count);
    for( size_t idx=0; idx < count; ++idx )
    {
        if( nulls && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(list, idx, Py_None);
            continue;
        }
        PyObject* pyval = PyLong_FromLong((long)results[idx]);
        PyList_SetItem(list, idx, pyval);
    }
    delete results;
    delete nulls;
    return list;
}

struct get_indexes_for_functor : base_functor
{
    get_indexes_for_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T>
    size_t operator()(PyObject* pykey, int* result)
    {
        size_t count;
        numeric_category<T>* this_ptr = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        if( pykey == Py_None )
        {
            Py_BEGIN_ALLOW_THREADS
            count = this_ptr->get_indexes_for_null_key(result);
            Py_END_ALLOW_THREADS
        }
        else
        {
            Py_BEGIN_ALLOW_THREADS
            T key = pyobj_convert<T>(pykey);
            count = this_ptr->get_indexes_for(key,result);
            Py_END_ALLOW_THREADS
        }
        return count;
    }
};

PyObject* ncat_get_indexes_for_key( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pykey = PyTuple_GetItem(args,1);
    PyObject* pyoutput = PyTuple_GetItem(args,2);
    if( pyoutput == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"get_indexes_for_key: output buffer of type int32 required");
        Py_RETURN_NONE;
    }
    DataHandler data(pyoutput,"int32");
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"get_indexes_for_key: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = tptr->get_type_name();
    // rather than check the type, we use get_values to ensure
    // dev-memory and then copy the results at the end
    // this is only wasteful if we are passed host memory
    int* results = reinterpret_cast<int*>(data.get_values());
    // the GIL guards are inside the functor above
    size_t count = type_dispatcher( tptr->get_type_name(), get_indexes_for_functor(tptr), pykey, results );
    if( !data.is_device_type() )
        data.results_to_host();
    //
    return PyLong_FromLong(count);
}

struct to_type_functor : base_functor
{
    to_type_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void operator()(void* items, BYTE* nulls)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        T* results = reinterpret_cast<T*>(items);
        cthis->to_type(results,nulls);
    }
};

PyObject* ncat_to_type( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pydata = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler data(pydata,tptr->get_type_name());
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"to_type %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->get_type_name(), size_functor(tptr) );
    if( count > data.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld %s values", count, tptr->get_type_name() );
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    Py_BEGIN_ALLOW_THREADS
    type_dispatcher( tptr->get_type_name(), to_type_functor(tptr), data.get_values(), reinterpret_cast<BYTE*>(nulls.get_values()) );
    if( !data.is_device_type() )
        data.results_to_host();
    if( !nulls.is_device_type() )
        nulls.results_to_host();
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

struct gather_type_functor : base_functor
{
    gather_type_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void operator()(int* indexes, unsigned int count, void* items, BYTE* nulls)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        T* results = reinterpret_cast<T*>(items);
        cthis->gather_type(indexes,count,results,nulls);
    }
};

PyObject* ncat_gather_type( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    PyObject* pydata = PyTuple_GetItem(args,2);
    PyObject* pynulls = PyTuple_GetItem(args,3);
    DataHandler indexes(pyindexes,"int32");
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"indexes %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    DataHandler data(pydata,tptr->get_type_name());
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"output %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    if( data.get_count() < indexes.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld %s values", indexes.get_count(), tptr->get_type_name() );
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    Py_BEGIN_ALLOW_THREADS
    type_dispatcher( tptr->get_type_name(), gather_type_functor(tptr),
                     reinterpret_cast<int*>(indexes.get_values()), indexes.get_count(),
                     data.get_values(), reinterpret_cast<BYTE*>(nulls.get_values()) );
    if( !data.is_device_type() )
        data.results_to_host();
    if( !nulls.is_device_type() )
        nulls.results_to_host();
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

struct gather_functor : base_functor
{
    gather_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(int* indexes, unsigned int count)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        auto result = cthis->gather(indexes,count);
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

PyObject* ncat_gather( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    DataHandler indexes(pyindexes,"int32");
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"gather %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    unsigned int count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
    if( count==0 )
        count = indexes.get_count();
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), gather_functor(tptr),
                              reinterpret_cast<int*>(indexes.get_values()), count);
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr(result);
}

struct gather_values_functor : base_functor
{
    gather_values_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(int* indexes, unsigned int count)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        auto result = cthis->gather_values(indexes,count);
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

PyObject* ncat_gather_values( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    DataHandler indexes(pyindexes,"int32");
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"gather_values %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    unsigned int count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
    if( count==0 )
        count = indexes.get_count();
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), gather_values_functor(tptr),
                              reinterpret_cast<int*>(indexes.get_values()), count);
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

struct gather_remap_functor : base_functor
{
    gather_remap_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(int* indexes, unsigned int count)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        auto result = cthis->gather_and_remap(indexes,count);
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

PyObject* ncat_gather_and_remap( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    DataHandler indexes(pyindexes,"int32");
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"gather_and_remap %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    unsigned int count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
    if( count==0 )
        count = indexes.get_count();
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), gather_remap_functor(tptr),
                              reinterpret_cast<int*>(indexes.get_values()), count);
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

struct add_keys_functor : base_functor
{
    add_keys_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(void* keys, unsigned int count, BYTE* nulls)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        auto result = cthis->add_keys(reinterpret_cast<T*>(keys),count,nulls);
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

PyObject* ncat_add_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pykeys = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler keys(pykeys,tptr->get_type_name());
    if( keys.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"add_keys: indexes %s", keys.get_error_text());
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), add_keys_functor(tptr),
                              keys.get_values(), keys.get_count(),
                              reinterpret_cast<BYTE*>(nulls.get_values()) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

struct remove_keys_functor : base_functor
{
    remove_keys_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(void* keys, unsigned int count, BYTE* nulls)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        auto result = cthis->remove_keys(reinterpret_cast<T*>(keys),count,nulls);
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

PyObject* ncat_remove_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pykeys = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler keys(pykeys,tptr->get_type_name());
    if( keys.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"remove_keys: %s", keys.get_error_text());
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), remove_keys_functor(tptr),
                              keys.get_values(), keys.get_count(),
                              reinterpret_cast<BYTE*>(nulls.get_values()) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

struct remove_unused_functor : base_functor
{
    remove_unused_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()()
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        auto result = cthis->remove_unused_keys();
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

PyObject* ncat_remove_unused( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), remove_unused_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

struct set_keys_functor : base_functor
{
    set_keys_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(void* keys, unsigned int count, BYTE* nulls)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        auto result = cthis->set_keys(reinterpret_cast<T*>(keys),count,nulls);
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

PyObject* ncat_set_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pykeys = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler keys(pykeys,tptr->get_type_name());
    if( keys.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"set_keys: %s", keys.get_error_text());
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), set_keys_functor(tptr),
                              keys.get_values(), keys.get_count(),
                              reinterpret_cast<BYTE*>(nulls.get_values()) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

struct merge_functor : base_functor
{
    merge_functor(base_category_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(void* cat)
    {
        numeric_category<T>* cthis = reinterpret_cast<numeric_category<T>*>(obj_ptr);
        numeric_category<T>* cthat = reinterpret_cast<numeric_category<T>*>(cat);
        auto result = cthis->merge(*cthat);
        //result->print();
        return reinterpret_cast<void*>(result);
    }
};

PyObject* ncat_merge_category( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pycat = PyTuple_GetItem(args,1);
    if( pycat == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"merge: argument cannot be null");
        Py_RETURN_NONE;
    }
    //
    std::string cname = pycat->ob_type->tp_name;
    if( cname.compare("nvcategory")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"argument must be nvcategory object");
        Py_RETURN_NONE;
    }
    base_category_type* tcat = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyObject_GetAttrString(pycat,"m_cptr")));
    std::string tname = tptr->get_type_name();
    if( tname.compare(tcat->get_type_name())!=0 )
    {
        PyErr_Format(PyExc_ValueError,"argument category type %s does not match target type %s", tcat->get_type_name(), tname.c_str() );
        Py_RETURN_NONE;
    }
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), merge_functor(tptr), tcat );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

