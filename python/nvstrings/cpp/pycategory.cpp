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

#include <Python.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>
#include <nvstrings/numeric_category.h>

#include "./numeric_category.h"

const char* string_type_name = "custring";

static PyObject* n_createCategoryFromNVStrings( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0);
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: parameter required");
        Py_RETURN_NONE;
    }
    std::vector<NVStrings*> strslist;
    // parameter can be a list of nvstrings instances
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pystrs);
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pystr = PyList_GetItem(pystrs,idx);
            cname = pystr->ob_type->tp_name;
            if( cname.compare("nvstrings")!=0 )
            {
                PyErr_Format(PyExc_ValueError,"nvcategory: argument list must contain nvstrings objects");
                Py_RETURN_NONE;
            }
            NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystr,"m_cptr"));
            if( strs==0 )
            {
                PyErr_Format(PyExc_ValueError,"nvcategory: invalid nvstrings object");
                Py_RETURN_NONE;
            }
            strslist.push_back(strs);
        }
    }
    // or a single nvstrings instance
    else if( cname.compare("nvstrings")==0 )
    {
        NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
        if( strs==0 )
        {
            PyErr_Format(PyExc_ValueError,"nvcategory: invalid nvstrings object");
            Py_RETURN_NONE;
        }
        strslist.push_back(strs);
    }
    else
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: argument must be nvstrings object");
        Py_RETURN_NONE;
    }

    NVCategory* thisptr = nullptr;
    Py_BEGIN_ALLOW_THREADS
    thisptr = NVCategory::create_from_strings(strslist);
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)thisptr);
}

// called by to_device() method in python class
static PyObject* n_createCategoryFromHostStrings( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0); // only one parm expected

    // handle single string
    if( PyObject_TypeCheck(pystrs,&PyUnicode_Type) )
    {
        const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));
        return PyLong_FromVoidPtr((void*)NVCategory::create_from_array(&str,1));
    }

    // check for list type
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("list")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: argument must be a list of strings");
        Py_RETURN_NONE;
    }

    // handle array of strings
    unsigned int count = (unsigned int)PyList_Size(pystrs);
    const char** list = new const char*[count];
    for( unsigned int idx=0; idx < count; ++idx )
    {
        PyObject* pystr = PyList_GetItem(pystrs,idx);
        if( (pystr == Py_None) || !PyObject_TypeCheck(pystr,&PyUnicode_Type) )
            list[idx] = 0;
        else
            list[idx] = PyUnicode_AsUTF8(pystr);
    }
    //
    NVCategory* thisptr = nullptr;
    Py_BEGIN_ALLOW_THREADS
    thisptr = NVCategory::create_from_array(list,count);
    Py_END_ALLOW_THREADS
    delete list;
    return PyLong_FromVoidPtr((void*)thisptr);
}

// called by from_offsets() method in python class
static PyObject* n_createFromOffsets( PyObject* self, PyObject* args )
{
    PyObject* pysbuf = PyTuple_GetItem(args,0);
    PyObject* pyobuf = PyTuple_GetItem(args,1);
    PyObject* pyscount = PyTuple_GetItem(args,2);
    PyObject* pynbuf = PyTuple_GetItem(args,3);
    PyObject* pyncount = PyTuple_GetItem(args,4);

    //
    if( (pysbuf == Py_None) || (pyobuf == Py_None) )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: missing parameter");
        Py_RETURN_NONE;
    }

    const char* sbuffer = 0;
    const int* obuffer = 0;
    const unsigned char* nbuffer = 0;
    unsigned int scount = (unsigned int)PyLong_AsLong(pyscount);
    unsigned int ncount = 0;

    Py_buffer sbuf, obuf, nbuf;
    if( PyObject_CheckBuffer(pysbuf) )
    {
        PyObject_GetBuffer(pysbuf,&sbuf,PyBUF_SIMPLE);
        sbuffer = (const char*)sbuf.buf;
    }
    else
        sbuffer = (const char*)PyLong_AsVoidPtr(pysbuf);

    if( PyObject_CheckBuffer(pyobuf) )
    {
        PyObject_GetBuffer(pyobuf,&obuf,PyBUF_SIMPLE);
        obuffer = (const int*)obuf.buf;
    }
    else
        obuffer = (const int*)PyLong_AsVoidPtr(pyobuf);

    if( PyObject_CheckBuffer(pynbuf) )
    {
        PyObject_GetBuffer(pynbuf,&nbuf,PyBUF_SIMPLE);
        nbuffer = (const unsigned char*)nbuf.buf;
    }
    else if( pynbuf != Py_None )
    {
        nbuffer = (const unsigned char*)PyLong_AsVoidPtr(pynbuf);
        ncount = (unsigned int)PyLong_AsLong(pyncount);
    }

    PyObject* pybmem = PyTuple_GetItem(args,5);
    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    //printf(" ptrs=%p,%p,%p\n",sbuffer,obuffer,nbuffer);
    //printf(" scount=%d,ncount=%d\n",scount,ncount);
    // create strings object from these buffers
    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = NVCategory::create_from_offsets(sbuffer,scount,obuffer,nbuffer,ncount,bdevmem);
    Py_END_ALLOW_THREADS

    if( PyObject_CheckBuffer(pysbuf) )
        PyBuffer_Release(&sbuf);
    if( PyObject_CheckBuffer(pyobuf) )
        PyBuffer_Release(&obuf);
    if( PyObject_CheckBuffer(pynbuf) )
        PyBuffer_Release(&nbuf);

    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyObject* n_createCategoryFromNumbers( PyObject* self, PyObject* args )
{
    return ncat_createCategoryFromBuffer(self,args);
}


// called by destructor in python class
static PyObject* n_destroyCategory( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_destroyCategory(self,args);
    Py_BEGIN_ALLOW_THREADS
    NVCategory::destroy(reinterpret_cast<NVCategory*>(tptr));
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(0);
}

static PyObject* n_size( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_size(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    size_t count = cat->size();
    return PyLong_FromLong(count);
}

static PyObject* n_keys_size( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_keys_size(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    size_t count = cat->keys_size();
    return PyLong_FromLong(count);
}

static PyObject* n_get_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_get_keys(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    NVStrings* strs = nullptr;
    Py_BEGIN_ALLOW_THREADS
    strs = cat->get_keys();
    Py_END_ALLOW_THREADS
    if( strs )
        return PyLong_FromVoidPtr((void*)strs);
    Py_RETURN_NONE;
}

static PyObject* n_keys_type( PyObject* self, PyObject* args )
{
    base_category_type* tptr = reinterpret_cast<base_category_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    std::string type_name;
    Py_BEGIN_ALLOW_THREADS
    type_name = tptr->get_type_name();
    Py_END_ALLOW_THREADS
    return PyUnicode_FromString(type_name.c_str());
}

static PyObject* n_get_value_for_index( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
    {
        PyErr_Format(PyExc_ValueError,"method not implemented for this category type (%s)", tname.c_str() );
        Py_RETURN_NONE;
    }
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    unsigned int index = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,1));
    int value = 0;
    Py_BEGIN_ALLOW_THREADS
    value = cat->get_value(index);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(value);
}

static PyObject* n_get_value_for_string( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
    {
        PyErr_Format(PyExc_ValueError,"method not implemented for this category type (%s)", tname.c_str() );
        Py_RETURN_NONE;
    }
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* argStr = PyTuple_GetItem(args,1);
    int rtn = -1;
    const char* str = 0;
    if( argStr != Py_None )
        str = PyUnicode_AsUTF8(argStr);
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->get_value(str);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(rtn);
}

static PyObject* n_get_values( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_get_values(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    unsigned int count = cat->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        cat->get_values(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    cat->get_values(rtn,false);
    Py_END_ALLOW_THREADS
    for(unsigned idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_get_values_cpointer( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_values_cpointer(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    const int * vptr = nullptr;
    Py_BEGIN_ALLOW_THREADS
    vptr = cat->values_cptr();
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)vptr);
}

static PyObject* n_get_indexes_for_key( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_get_indexes_for_key(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* argStr = PyTuple_GetItem(args,1);
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    const char* str = 0;
    if( argStr != Py_None )
        str = PyUnicode_AsUTF8(argStr);
    if( devptr )
    {
        int count = 0;
        Py_BEGIN_ALLOW_THREADS
        count = cat->get_indexes_for(str,devptr);
        Py_END_ALLOW_THREADS
        if( count < 0 )
            PyErr_Format(PyExc_ValueError,"nvcategory: string not found in keys");
        return PyLong_FromLong(count);
    }
    //
    int count = 0;
    Py_BEGIN_ALLOW_THREADS
    count = cat->get_indexes_for(str,0,false);
    Py_END_ALLOW_THREADS
    if( count < 0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: string not found in keys");
        Py_RETURN_NONE;
    }
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    // copy to host option
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    cat->get_indexes_for(str,rtn,false);
    Py_END_ALLOW_THREADS
    for(int idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

static PyObject* n_add_strings( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
    {
        PyErr_Format(PyExc_ValueError,"invalid category type (%s) for this method", tname.c_str() );
        Py_RETURN_NONE;
    }
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pystrs = PyTuple_GetItem(args,1);
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.add_strings: parameter required");
        Py_RETURN_NONE;
    }
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("nvstrings")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.add_strings: argument must be nvstrings object");
        Py_RETURN_NONE;
    }
    NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
    if( strs==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.add_strings: invalid nvstrings object");
        Py_RETURN_NONE;
    }

    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->add_strings(*strs);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_remove_strings( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
    {
        PyErr_Format(PyExc_ValueError,"invalid category type (%s) for this method", tname.c_str() );
        Py_RETURN_NONE;
    }
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pystrs = PyTuple_GetItem(args,1);
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.remove_strings: parameter required");
        Py_RETURN_NONE;
    }
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("nvstrings")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.remove_strings: argument must be nvstrings object");
        Py_RETURN_NONE;
    }
    NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
    if( strs==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.remove_strings: invalid nvstrings object");
        Py_RETURN_NONE;
    }

    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->remove_strings(*strs);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_to_strings( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
    {
        PyErr_Format(PyExc_ValueError,"invalid category type (%s) for this method -- use to_numbers() instead", tname.c_str() );
        Py_RETURN_NONE;
    }
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    NVStrings* strs = nullptr;
    Py_BEGIN_ALLOW_THREADS
    strs = cat->to_strings();
    Py_END_ALLOW_THREADS
    if( strs )
        return PyLong_FromVoidPtr((void*)strs);
    Py_RETURN_NONE;
}

static PyObject* n_to_numbers( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name)==0 )
    {
        PyErr_Format(PyExc_ValueError,"invalid category type for this method -- use to_strings() instead");
        Py_RETURN_NONE;
    }
    return ncat_to_type(self,args);
}

static PyObject* n_gather_strings( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
    {
        PyErr_Format(PyExc_ValueError,"invalid category type (%s) for this method -- use gather_numbers() instead", tname.c_str() );
        Py_RETURN_NONE;
    }
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pyidxs = PyTuple_GetItem(args,1);
    std::string cname = pyidxs->ob_type->tp_name;
    NVStrings* rtn = 0;
    std::string message;
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pyidxs);
        int* indexes = new int[count];
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pyidx = PyList_GetItem(pyidxs,idx);
            indexes[idx] = (int)PyLong_AsLong(pyidx);
        }
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = cat->gather_strings(indexes,count,false);
        }
        catch(const std::out_of_range& eor)
        {
            std::ostringstream errmsg;
            errmsg << "one or more indexes out of range [0:" << cat->keys_size() << ")";
            message = errmsg.str();
        }
        Py_END_ALLOW_THREADS
        delete indexes;
    }
    else
    {
        // assume device pointer
        int* indexes = (int*)PyLong_AsVoidPtr(pyidxs);
        unsigned int count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = cat->gather_strings(indexes,count);
        }
        catch(const std::out_of_range& eor)
        {
            std::ostringstream errmsg;
            errmsg << "one or more indexes out of range [0:" << cat->keys_size() << ")";
            message = errmsg.str();
        }
        Py_END_ALLOW_THREADS
    }

    if( !message.empty() )
        PyErr_Format(PyExc_IndexError,message.c_str());
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_gather_numbers( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name)==0 )
    {
        PyErr_Format(PyExc_ValueError,"invalid category type for this method -- use gather_strings() instead");
        Py_RETURN_NONE;
    }
    return ncat_gather_type(self,args);
}

static PyObject* n_gather( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_gather(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pyidxs = PyTuple_GetItem(args,1);
    std::string cname = pyidxs->ob_type->tp_name;
    NVCategory* rtn = 0;
    std::string message;

    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pyidxs);
        int* indexes = new int[count];
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pyidx = PyList_GetItem(pyidxs,idx);
            indexes[idx] = (int)PyLong_AsLong(pyidx);
        }
        //
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = cat->gather(indexes,count,false);
        }
        catch(const std::out_of_range& eor)
        {
            std::ostringstream errmsg;
            errmsg << "one or more indexes out of range [0:" << cat->keys_size() << ")";
            message = errmsg.str();
        }
        Py_END_ALLOW_THREADS
        delete indexes;
    }
    else
    {
        // assume device pointer
        int* indexes = (int*)PyLong_AsVoidPtr(pyidxs);
        unsigned int count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = cat->gather(indexes,count);
        }
        catch(const std::out_of_range& eor)
        {
            std::ostringstream errmsg;
            errmsg << "one or more indexes out of range [0:" << cat->keys_size() << ")";
            message = errmsg.str();
        }
        Py_END_ALLOW_THREADS
    }

    if( !message.empty() )
        PyErr_Format(PyExc_IndexError,message.c_str());
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_gather_and_remap( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_gather_and_remap(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pyidxs = PyTuple_GetItem(args,1);
    std::string cname = pyidxs->ob_type->tp_name;
    NVCategory* rtn = 0;
    std::string message;

    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pyidxs);
        int* indexes = new int[count];
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pyidx = PyList_GetItem(pyidxs,idx);
            indexes[idx] = (int)PyLong_AsLong(pyidx);
        }
        //
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = cat->gather_and_remap(indexes,count,false);
        }
        catch(const std::out_of_range& eor)
        {
            std::ostringstream errmsg;
            errmsg << "one or more indexes out of range [0:" << cat->keys_size() << ")";
            message = errmsg.str();
        }
        Py_END_ALLOW_THREADS
        delete indexes;
    }
    else
    {
        // assume device pointer
        int* indexes = (int*)PyLong_AsVoidPtr(pyidxs);
        unsigned int count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = cat->gather_and_remap(indexes,count);
        }
        catch(const std::out_of_range& eor)
        {
            std::ostringstream errmsg;
            errmsg << "one or more indexes out of range [0:" << cat->keys_size() << ")";
            message = errmsg.str();
        }
        Py_END_ALLOW_THREADS
    }

    if( !message.empty() )
        PyErr_Format(PyExc_IndexError,message.c_str());
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_merge_category( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
    {
        PyErr_Format(PyExc_ValueError,"method not implemented for this category type (%s)", tname.c_str() );
        Py_RETURN_NONE;
    }
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pycat2 = PyTuple_GetItem(args,1);
    if( pycat2 == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.merge_category: parameter required");
        Py_RETURN_NONE;
    }
    std::string cname = pycat2->ob_type->tp_name;
    if( cname.compare("nvcategory")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.merge_category: argument must be nvcategory object");
        Py_RETURN_NONE;
    }
    NVCategory* cat2 = (NVCategory*)PyLong_AsVoidPtr(PyObject_GetAttrString(pycat2,"m_cptr"));
    if( cat2==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.merge_category: invalid nvcategory object");
        Py_RETURN_NONE;
    }

    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->merge_category(*cat2);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_merge_and_remap( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_merge_category(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pycat = PyTuple_GetItem(args,1);
    if( pycat == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.merge_and_remap: parameter required");
        Py_RETURN_NONE;
    }
    std::string cname = pycat->ob_type->tp_name;
    if( cname.compare("nvcategory")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.merge_and_remap: argument must be nvcategory object");
        Py_RETURN_NONE;
    }
    NVCategory* cat2 = (NVCategory*)PyLong_AsVoidPtr(PyObject_GetAttrString(pycat,"m_cptr"));
    if( cat2==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.merge_and_remap: invalid nvcategory object");
        Py_RETURN_NONE;
    }

    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->merge_and_remap(*cat2);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_add_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_add_keys(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pystrs = PyTuple_GetItem(args,1);
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.add_keys: parameter required");
        Py_RETURN_NONE;
    }
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("nvstrings")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.add_keys: argument must be nvstrings object");
        Py_RETURN_NONE;
    }
    NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
    if( strs==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.add_keys: invalid nvstrings object");
        Py_RETURN_NONE;
    }

    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->add_keys_and_remap(*strs);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_remove_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_remove_keys(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pystrs = PyTuple_GetItem(args,1);
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.remove_keys: parameter required");
        Py_RETURN_NONE;
    }
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("nvstrings")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.remove_keys: argument must be nvstrings object");
        Py_RETURN_NONE;
    }
    NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
    if( strs==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.remove_keys: invalid nvstrings object");
        Py_RETURN_NONE;
    }

    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->remove_keys_and_remap(*strs);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_remove_unused_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_remove_unused(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->remove_unused_keys_and_remap();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_set_keys( PyObject* self, PyObject* args )
{
    base_category_type* tptr = (base_category_type*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    std::string tname = tptr->get_type_name();
    if( tname.compare(string_type_name) )
        return ncat_set_keys(self,args);
    NVCategory* cat = reinterpret_cast<NVCategory*>(tptr);
    PyObject* pystrs = PyTuple_GetItem(args,1);
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.set_keys: parameter required");
        Py_RETURN_NONE;
    }
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("nvstrings")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.set_keys: argument must be nvstrings object");
        Py_RETURN_NONE;
    }
    NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
    if( strs==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory.set_keys: invalid nvstrings object");
        Py_RETURN_NONE;
    }

    NVCategory* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = cat->set_keys_and_remap(*strs);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyMethodDef s_Methods[] = {
    { "n_createCategoryFromHostStrings", n_createCategoryFromHostStrings, METH_VARARGS, "" },
    { "n_createCategoryFromNVStrings", n_createCategoryFromNVStrings, METH_VARARGS, "" },
    { "n_createCategoryFromNumbers", n_createCategoryFromNumbers, METH_VARARGS, "" },
    { "n_createFromOffsets", n_createFromOffsets, METH_VARARGS, "" },
    { "n_destroyCategory", n_destroyCategory, METH_VARARGS, "" },
    { "n_size", n_size, METH_VARARGS, "" },
    { "n_keys_size", n_keys_size, METH_VARARGS, "" },
    { "n_keys_type", n_keys_type, METH_VARARGS, "" },
    { "n_get_keys", n_get_keys, METH_VARARGS, "" },
    { "n_get_indexes_for_key", n_get_indexes_for_key, METH_VARARGS, "" },
    { "n_get_value_for_index", n_get_value_for_index, METH_VARARGS, "" },
    { "n_get_value_for_string", n_get_value_for_string, METH_VARARGS, "" },
    { "n_get_values", n_get_values, METH_VARARGS, "" },
    { "n_get_values_cpointer", n_get_values_cpointer, METH_VARARGS, "" },
    { "n_add_strings", n_add_strings, METH_VARARGS, "" },
    { "n_remove_strings", n_remove_strings, METH_VARARGS, "" },
    { "n_to_strings", n_to_strings, METH_VARARGS, "" },
    { "n_to_numbers", n_to_numbers, METH_VARARGS, "" },
    { "n_gather_strings", n_gather_strings, METH_VARARGS, "" },
    { "n_gather_numbers", n_gather_numbers, METH_VARARGS, "" },
    { "n_gather", n_gather, METH_VARARGS, "" },
    { "n_gather_and_remap", n_gather_and_remap, METH_VARARGS, "" },
    { "n_merge_category", n_merge_category, METH_VARARGS, "" },
    { "n_merge_and_remap", n_merge_and_remap, METH_VARARGS, "" },
    { "n_add_keys", n_add_keys, METH_VARARGS, "" },
    { "n_remove_keys", n_remove_keys, METH_VARARGS, "" },
    { "n_remove_unused_keys", n_remove_unused_keys, METH_VARARGS, "" },
    { "n_set_keys", n_set_keys, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem = {	PyModuleDef_HEAD_INIT, "NVCategory_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_pyniNVCategory(void)
{
    return PyModule_Create(&cModPyDem);
}
