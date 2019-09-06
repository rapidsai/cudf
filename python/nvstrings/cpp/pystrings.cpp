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
#include <map>
#include <string>
#include <stdio.h>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <nvstrings/NVStrings.h>
#include <nvstrings/ipc_transfer.h>
#include <nvstrings/StringsStatistics.h>

//
// These are C-functions that simply map the python objects to appropriate methods
// in the C++ NVStrings class. There should normally be a 1:1 mapping of
// python nvstrings class methods to C++ NVStrings class method.
// The C-functions here handle marshalling the python object data to/from C/C++
// data structures.
//
// Some cooperation is here around host memory where memory may be freed here
// that is allocated inside the C++ class. This should probably be corrected
// since they may use different allocators and could be risky down the line.
//

// we handle alot of different data inputs
// this class handles them all and cleans them up appropriately
template<typename T>
class DataBuffer
{
    PyObject* pyobj;
    void* pdata;
    std::string name;
    enum listtype { none, error, blist, list, device_ndarray, ndarray, buffer, pointer };
    listtype ltype;
    std::string errortext;
    unsigned int type_width;
    std::string dtype_name;

    T* values;
    unsigned int count;

public:
    //
    DataBuffer( PyObject* obj ) : pyobj(obj), pdata(0), values(0), count(0), ltype(none)
    {
        type_width = sizeof(T);
        if( pyobj == Py_None )
            return;

        name = pyobj->ob_type->tp_name;
        if( name.compare("list")==0 )
        {
            count = (unsigned int)PyList_Size(pyobj);
            std::string stname = (count>0 ? PyList_GetItem(pyobj,0)->ob_type->tp_name : "");
            bool btype = (count>0) && (stname.compare("bool")==0);
            T* data = new T[count];
            for( unsigned int idx=0; idx < count; ++idx )
            {
                PyObject* pyidx = PyList_GetItem(pyobj,idx);
                if( pyidx == Py_None )
                    data[idx] = 0;
                else if( std::is_same<T,bool>::value )
                    data[idx] = PyObject_IsTrue(pyidx);
                else
                    data[idx] = (T)PyLong_AsLong(pyidx);
                stname = pyidx->ob_type->tp_name;
                btype &= stname.compare("bool")==0;
            }
            ltype = btype ? blist : list;
            values = data;
            pdata = data;
        }
        else if( name.compare("DeviceNDArray")==0 )
        {
            ltype = device_ndarray;
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
                values = (T*)PyLong_AsVoidPtr(pyobj);
                // get the dtype name in case that helps with type-checking
                dtype_name = PyUnicode_AsUTF8(PyObject_Str(pydtype));
            }
        }
        else if( name.compare("numpy.ndarray")==0 )
        {
            ltype = ndarray;
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
                values = (T*)PyLong_AsVoidPtr(pyobj);
                // get the dtype name in case that helps with type-checking
                dtype_name = PyUnicode_AsUTF8(PyObject_Str(pydtype));
            }
        }
        else if( PyObject_CheckBuffer(pyobj) )
        {
            ltype = buffer;
            Py_buffer* pybuf = new Py_buffer;
            PyObject_GetBuffer(pyobj,pybuf,PyBUF_SIMPLE);
            values = (T*)pybuf->buf;
            count = (unsigned int)(pybuf->len/sizeof(T));
            pdata = pybuf;
        }
        else if( name.compare("int")==0 )
        {
            ltype = pointer;
            values = (T*)PyLong_AsVoidPtr(pyobj);
        }
        else
        {
            ltype = error;
            errortext = "unknown_type: ";
            errortext += name;
        }
    }

    //
    ~DataBuffer()
    {
        if( ltype==list || ltype==blist )
            delete (T*)pdata;
        else if( ltype==buffer )
        {
            PyBuffer_Release((Py_buffer*)pdata);
            delete (Py_buffer*)pdata;
        }
    }

    //
    bool is_error()               { return ltype==error; }
    const char* get_error_text()  { return errortext.c_str(); }
    bool is_blist()               { return ltype==blist || (dtype_name.compare("bool")==0); }
    const char* get_name()        { return name.c_str(); }
    bool is_device_type()         { return (ltype==device_ndarray) || (ltype==pointer); }

    T* get_values()               { return values; }
    unsigned int get_count()      { return count; }
    unsigned int get_type_width() { return type_width; }
    const char* get_dtype_name()  { return dtype_name.c_str(); }
};

// PyArg_VaParse format types are documented here:
// https://docs.python.org/3/c-api/arg.html
bool parse_args( const char* fn, PyObject* pyargs, const char* pyfmt, ... )
{
    va_list args;
    va_start(args,pyfmt);
    bool rtn = (bool)PyArg_VaParse(pyargs,pyfmt,args);
    va_end(args);
    if( !rtn )
        PyErr_Format(PyExc_ValueError,"nvstrings.%s: invalid parameters",fn);
    return rtn;
}

static PyObject* n_createFromIPC( PyObject* self, PyObject* args )
{
    nvstrings_ipc_transfer ipc;
    memcpy(&ipc,PyByteArray_AsString(PyTuple_GetItem(args,0)),sizeof(nvstrings_ipc_transfer));
    NVStrings* thisptr = nullptr;
    Py_BEGIN_ALLOW_THREADS
    thisptr = NVStrings::create_from_ipc(ipc);
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)thisptr);
}

static PyObject* n_getIPCData( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    nvstrings_ipc_transfer ipc;

    Py_BEGIN_ALLOW_THREADS
    tptr->create_ipc_transfer(ipc);
    Py_END_ALLOW_THREADS
    return PyByteArray_FromStringAndSize((char*)&ipc,sizeof(ipc));
}

// called by to_device() method in python class
static PyObject* n_createFromHostStrings( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0); // only one parm expected

    // handle single string
    if( PyObject_TypeCheck(pystrs,&PyUnicode_Type) )
    {
        const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));
        NVStrings* thisptr = nullptr;
        Py_BEGIN_ALLOW_THREADS
        thisptr = NVStrings::create_from_array(&str,1);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)thisptr);
    }

    // would be cool if we could check the type is list/array
    //if( !PyObject_TypeCheck(pystrs, &PyArray_Type) )
    //    return PyLong_FromVoidPtr(0); // probably should throw exception

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
    //printf("creating %d strings in device memory\n",count);
    NVStrings* thisptr = nullptr;
    Py_BEGIN_ALLOW_THREADS
    thisptr = NVStrings::create_from_array(list,count);
    Py_END_ALLOW_THREADS
    delete list;
    return PyLong_FromVoidPtr((void*)thisptr);
}

// called by destructor in python class
static PyObject* n_destroyStrings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    Py_BEGIN_ALLOW_THREADS
    NVStrings::destroy(tptr);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(0);
}

// called in cases where the host code will want the strings back from the device
static PyObject* n_createHostStrings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);

    std::vector<char*> list(count);
    char** plist = list.data();
    std::vector<int> lens(count);
    size_t totalmem = tptr->byte_count(lens.data(),false);
    std::vector<char> buffer(totalmem+count,0); // null terminates each string
    char* pbuffer = buffer.data();
    size_t offset = 0;
    for( int idx=0; idx < count; ++idx )
    {
        plist[idx] = pbuffer + offset;
        offset += lens[idx]+1; // account for null-terminator; also nulls are -1
    }
    Py_BEGIN_ALLOW_THREADS
    tptr->to_host(plist,0,count);
    Py_END_ALLOW_THREADS
    PyObject* ret = PyList_New(count);
    for( unsigned int idx=0; idx < count; ++idx )
    {
        char* str = list[idx];
        if( lens[idx]>=0 )
        {
            PyList_SetItem(ret, idx, PyUnicode_FromString((const char*)str));
        }
        else
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
    }
    return ret;
}

static PyObject* n_createFromNVStrings( PyObject* self, PyObject* args )
{
    PyObject* pystrs = PyTuple_GetItem(args,0); // only one parm expected
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: parameter required");
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
                PyErr_Format(PyExc_ValueError,"nvstrings: argument list must contain nvstrings objects");
                Py_RETURN_NONE;
            }
            NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystr,"m_cptr"));
            if( strs==0 )
            {
                PyErr_Format(PyExc_ValueError,"nvstrings: invalid nvstrings object");
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
            PyErr_Format(PyExc_ValueError,"nvstrings: invalid nvstrings object");
            Py_RETURN_NONE;
        }
        strslist.push_back(strs);
    }
    else
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: argument must be nvstrings object");
        Py_RETURN_NONE;
    }

    NVStrings* thisptr = nullptr;
    Py_BEGIN_ALLOW_THREADS
    thisptr = NVStrings::create_from_strings(strslist);
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)thisptr);
}

// just for testing and should be removed
static PyObject* n_createFromCSV( PyObject* self, PyObject* args )
{
    std::string csvfile = PyUnicode_AsUTF8(PyTuple_GetItem(args,0));
    unsigned int column = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,1));
    unsigned int lines = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
    unsigned int flags = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,3));
    NVStrings::sorttype stype = (NVStrings::sorttype)(flags & 3);
    bool nullIsEmpty = (flags & 8) > 0;
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = NVStrings::create_from_csv(csvfile.c_str(),column,lines,stype,nullIsEmpty);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
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
        PyErr_Format(PyExc_ValueError,"nvstrings: missing parameter");
        Py_RETURN_NONE;
    }

    const char* sbuffer = 0;
    const int* obuffer = 0;
    const unsigned char* nbuffer = 0;
    int scount = (int)PyLong_AsLong(pyscount);
    int ncount = 0;

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
        ncount = (int)PyLong_AsLong(pyncount);
    }

    PyObject* pybmem = PyTuple_GetItem(args,5);
    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    //printf(" ptrs=%p,%p,%p\n",sbuffer,obuffer,nbuffer);
    //printf(" scount=%d,ncount=%d\n",scount,ncount);
    //printf(" bdevmem=%d\n",(int)bdevmem);
    // create strings object from these buffers
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = NVStrings::create_from_offsets(sbuffer,scount,obuffer,
                                         nbuffer,ncount,bdevmem);
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

static PyObject* n_createFromInt32s( PyObject* self, PyObject* args )
{
    PyObject* pyvals = PyTuple_GetItem(args,0);
    PyObject* pycount = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    PyObject* pybmem = PyTuple_GetItem(args,3);

    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    DataBuffer<int> dbvalues(pyvals);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.itos(): %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( dbvalues.get_type_width()!=sizeof(int) )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.itos(): values must be of type int32");
        Py_RETURN_NONE;
    }

    int* values = dbvalues.get_values();
    unsigned int count = dbvalues.get_count();
    if( count==0 )
        count = (unsigned int)PyLong_AsLong(pycount);

    NVStrings* rtn = 0;
    if( pynulls == Py_None )
    {
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::itos(values,count,0,bdevmem);
        Py_END_ALLOW_THREADS
    }
    else
    {   // get the nulls
        DataBuffer<unsigned char> dbnulls(pynulls);
        if( dbnulls.is_error() )
        {
            PyErr_Format(PyExc_TypeError,"nvstrings.itos(): %s",dbnulls.get_error_text());
            Py_RETURN_NONE;
        }
        unsigned char* nulls = dbnulls.get_values();
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::itos(values,count,nulls,bdevmem);
        Py_END_ALLOW_THREADS
    }
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_createFromInt64s( PyObject* self, PyObject* args )
{
    PyObject* pyvals = PyTuple_GetItem(args,0);
    PyObject* pycount = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    PyObject* pybmem = PyTuple_GetItem(args,3);

    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    DataBuffer<long> dbvalues(pyvals);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.ltos(): %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( dbvalues.get_type_width()!=sizeof(long) )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.ltos(): values must be of type int64");
        Py_RETURN_NONE;
    }

    long* values = dbvalues.get_values();
    unsigned int count = dbvalues.get_count();
    if( count==0 )
        count = (unsigned int)PyLong_AsLong(pycount);

    NVStrings* rtn = 0;
    if( pynulls == Py_None )
    {
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::ltos(values,count,0,bdevmem);
        Py_END_ALLOW_THREADS
    }
    else
    {   // get the nulls
        DataBuffer<unsigned char> dbnulls(pynulls);
        if( dbnulls.is_error() )
        {
            PyErr_Format(PyExc_TypeError,"nvstrings.ltos(): %s",dbnulls.get_error_text());
            Py_RETURN_NONE;
        }
        unsigned char* nulls = dbnulls.get_values();
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::ltos(values,count,nulls,bdevmem);
        Py_END_ALLOW_THREADS
    }
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_createFromFloat32s( PyObject* self, PyObject* args )
{
    PyObject* pyvals = PyTuple_GetItem(args,0);
    PyObject* pycount = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    PyObject* pybmem = PyTuple_GetItem(args,3);

    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    DataBuffer<float> dbvalues(pyvals);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.ftos(): %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( dbvalues.get_type_width()!=sizeof(float) )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.ftos(): values must be of type float32");
        Py_RETURN_NONE;
    }

    float* values = dbvalues.get_values();
    unsigned int count = dbvalues.get_count();
    if( count==0 )
        count = (unsigned int)PyLong_AsLong(pycount);

    NVStrings* rtn = 0;
    if( pynulls == Py_None )
    {
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::ftos(values,count,0,bdevmem);
        Py_END_ALLOW_THREADS
    }
    else
    {   // get the nulls
        DataBuffer<unsigned char> dbnulls(pynulls);
        if( dbnulls.is_error() )
        {
            PyErr_Format(PyExc_TypeError,"nvstrings.ftos(): %s",dbnulls.get_error_text());
            Py_RETURN_NONE;
        }
        unsigned char* nulls = dbnulls.get_values();
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::ftos(values,count,nulls,bdevmem);
        Py_END_ALLOW_THREADS
    }
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_createFromFloat64s( PyObject* self, PyObject* args )
{
    PyObject* pyvals = PyTuple_GetItem(args,0);
    PyObject* pycount = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    PyObject* pybmem = PyTuple_GetItem(args,3);

    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    DataBuffer<double> dbvalues(pyvals);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.dtos(): %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( dbvalues.get_type_width()!=sizeof(double) )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.dtos(): values must be of type float64");
        Py_RETURN_NONE;
    }

    double* values = dbvalues.get_values();
    unsigned int count = dbvalues.get_count();
    if( count==0 )
        count = (unsigned int)PyLong_AsLong(pycount);

    NVStrings* rtn = 0;
    if( pynulls == Py_None )
    {
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::dtos((double*)values,count,0,bdevmem);
        Py_END_ALLOW_THREADS
    }
    else
    {   // get the nulls
        DataBuffer<unsigned char> dbnulls(pynulls);
        if( dbnulls.is_error() )
        {
            PyErr_Format(PyExc_TypeError,"nvstrings.ftos(): %s",dbnulls.get_error_text());
            Py_RETURN_NONE;
        }
        unsigned char* nulls = dbnulls.get_values();
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::dtos((double*)values,count,nulls,bdevmem);
        Py_END_ALLOW_THREADS
    }
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_createFromIPv4Integers( PyObject* self, PyObject* args )
{
    PyObject* pyvals = PyTuple_GetItem(args,0);
    PyObject* pycount = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    PyObject* pybmem = PyTuple_GetItem(args,3);

    bool bdevmem = (bool)PyObject_IsTrue(pybmem);
    DataBuffer<unsigned int> dbvalues(pyvals);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.int2ip(): %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( dbvalues.get_type_width()!=sizeof(int) )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.int2ip(): values must be of type int32");
        Py_RETURN_NONE;
    }

    unsigned int* values = dbvalues.get_values();
    unsigned int count = dbvalues.get_count();
    if( count==0 )
        count = (unsigned int)PyLong_AsLong(pycount);
    //bdevmem = dbvalues.is_device_type();

    NVStrings* rtn = 0;
    if( pynulls == Py_None )
    {
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::int2ip(values,count,0,bdevmem);
        Py_END_ALLOW_THREADS
    }
    else
    {   // get the nulls
        DataBuffer<unsigned char> dbnulls(pynulls);
        if( dbnulls.is_error() )
        {
            PyErr_Format(PyExc_TypeError,"nvstrings.int2ip(): %s",dbnulls.get_error_text());
            Py_RETURN_NONE;
        }
        unsigned char* nulls = dbnulls.get_values();
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::int2ip(values,count,nulls,bdevmem);
        Py_END_ALLOW_THREADS
    }
    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// map parameter string to units
std::map<std::string,NVStrings::timestamp_units> name_units = {
    {"Y",NVStrings::timestamp_units::years}, {"M",NVStrings::timestamp_units::months}, {"D",NVStrings::timestamp_units::days},
    {"h",NVStrings::timestamp_units::hours}, {"m",NVStrings::timestamp_units::minutes}, {"s",NVStrings::timestamp_units::seconds},
    {"ms",NVStrings::timestamp_units::ms}, {"us",NVStrings::timestamp_units::us}, {"ns",NVStrings::timestamp_units::ns} };

static PyObject* n_createFromTimestamp( PyObject* self, PyObject* args )
{
    PyObject* pyvals = PyTuple_GetItem(args,0);
    PyObject* pycount = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    PyObject* argFormat = PyTuple_GetItem(args,3);
    PyObject* argUnits = PyTuple_GetItem(args,4);

    const char* format = nullptr;
    if( argFormat != Py_None )
        format = PyUnicode_AsUTF8(argFormat);

    const char* unitsz = PyUnicode_AsUTF8(argUnits);
    std::string str_units = unitsz;

    if( name_units.find(str_units)==name_units.end() )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: units parameter value unrecognized");
        Py_RETURN_NONE;
    }

    NVStrings::timestamp_units units = (NVStrings::timestamp_units)name_units[str_units];
    PyObject* pybmem = PyTuple_GetItem(args,5);
    bool bdevmem = (bool)PyObject_IsTrue(pybmem);
    DataBuffer<unsigned long> dbvalues(pyvals);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.int2timestamp(): %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( dbvalues.get_type_width()!=sizeof(long) )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.int2timestamp(): values must be of type int64");
        Py_RETURN_NONE;
    }

    unsigned long* values = dbvalues.get_values();
    unsigned int count = dbvalues.get_count();
    if( count==0 )
        count = (unsigned int)PyLong_AsLong(pycount);
    //bdevmem = dbvalues.is_device_type();

    NVStrings* rtn = 0;
    if( pynulls == Py_None )
    {
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::long2timestamp(values,count,units,format,0,bdevmem);
        Py_END_ALLOW_THREADS
    }
    else
    {   // get the nulls
        DataBuffer<unsigned char> dbnulls(pynulls);
        if( dbnulls.is_error() )
        {
            PyErr_Format(PyExc_TypeError,"nvstrings.int2timestamp(): %s",dbnulls.get_error_text());
            Py_RETURN_NONE;
        }
        unsigned char* nulls = dbnulls.get_values();
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::long2timestamp(values,count,units,format,nulls,bdevmem);
        Py_END_ALLOW_THREADS
    }

    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_createFromBools( PyObject* self, PyObject* args )
{
    PyObject* pyvals = PyTuple_GetItem(args,0);
    PyObject* pycount = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    PyObject* pytstr = PyTuple_GetItem(args,3);
    PyObject* pyfstr = PyTuple_GetItem(args,4);
    PyObject* pybmem = PyTuple_GetItem(args,5);
    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    DataBuffer<bool> dbvalues(pyvals);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.from_bools(): %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( pytstr==Py_None )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.from_bools(): true must not be null");
        Py_RETURN_NONE;
    }
    const char* tstr = PyUnicode_AsUTF8(pytstr);
    if( pyfstr==Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.from_bools(): false must not be null");
        Py_RETURN_NONE;
    }
    const char* fstr = PyUnicode_AsUTF8(pyfstr);

    bool* values = dbvalues.get_values();
    unsigned int count = dbvalues.get_count();
    if( count==0 )
        count = (unsigned int)PyLong_AsLong(pycount);

    NVStrings* rtn = 0;
    if( pynulls == Py_None )
    {
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::create_from_bools(values,count,tstr,fstr,0,bdevmem);
        Py_END_ALLOW_THREADS
    }
    else
    {   // get the nulls
        DataBuffer<unsigned char> dbnulls(pynulls);
        if( dbnulls.is_error() )
        {
            PyErr_Format(PyExc_TypeError,"nvstrings.from_bools(): %s",dbnulls.get_error_text());
            Py_RETURN_NONE;
        }
        unsigned char* nulls = dbnulls.get_values();
        Py_BEGIN_ALLOW_THREADS
        rtn = NVStrings::create_from_bools(values,count,tstr,fstr,nulls,bdevmem);
        Py_END_ALLOW_THREADS
    }
    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// called by from_offsets() method in python class
static PyObject* n_create_offsets( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pysbuf = PyTuple_GetItem(args,1);
    PyObject* pyobuf = PyTuple_GetItem(args,2);
    PyObject* pynbuf = PyTuple_GetItem(args,3);

    //
    if( (pysbuf == Py_None) || (pyobuf == Py_None) )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: missing parameter");
        Py_RETURN_NONE;
    }

    char* sbuffer = 0;
    int* obuffer = 0;
    unsigned char* nbuffer = 0;

    Py_buffer sbuf, obuf, nbuf;
    if( PyObject_CheckBuffer(pysbuf) )
    {
        PyObject_GetBuffer(pysbuf,&sbuf,PyBUF_SIMPLE);
        sbuffer = (char*)sbuf.buf;
    }
    else
        sbuffer = (char*)PyLong_AsVoidPtr(pysbuf);

    if( PyObject_CheckBuffer(pyobuf) )
    {
        PyObject_GetBuffer(pyobuf,&obuf,PyBUF_SIMPLE);
        obuffer = (int*)obuf.buf;
    }
    else
        obuffer = (int*)PyLong_AsVoidPtr(pyobuf);

    if( PyObject_CheckBuffer(pynbuf) )
    {
        PyObject_GetBuffer(pynbuf,&nbuf,PyBUF_SIMPLE);
        nbuffer = (unsigned char*)nbuf.buf;
    }
    else if( pynbuf != Py_None )
        nbuffer = (unsigned char*)PyLong_AsVoidPtr(pynbuf);

    PyObject* pybmem = PyTuple_GetItem(args,4);
    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    // create strings object from these buffers
    Py_BEGIN_ALLOW_THREADS
    tptr->create_offsets(sbuffer,obuffer,nbuffer,bdevmem);
    Py_END_ALLOW_THREADS

    if( PyObject_CheckBuffer(pysbuf) )
        PyBuffer_Release(&sbuf);
    if( PyObject_CheckBuffer(pyobuf) )
        PyBuffer_Release(&obuf);
    if( PyObject_CheckBuffer(pynbuf) )
        PyBuffer_Release(&nbuf);

    Py_RETURN_NONE;
}

static PyObject* n_size( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    return PyLong_FromLong(count);
}

// return the length of each string
static PyObject* n_len( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->len(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->len(rtn,false);
    Py_END_ALLOW_THREADS
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < 0 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

static PyObject* n_byte_count( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int* memptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    bool bdevmem = (bool)PyObject_IsTrue(PyTuple_GetItem(args,2));
    size_t rtn = 0;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->byte_count(memptr,bdevmem);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong((long)rtn);
}

// return the number of nulls
static PyObject* n_null_count( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool ben = (bool)PyObject_IsTrue(PyTuple_GetItem(args,1));
    unsigned int nulls = 0;
    Py_BEGIN_ALLOW_THREADS
    nulls = tptr->get_nulls(0,ben,false);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong((long)nulls);
}

static PyObject* n_set_null_bitmask( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pynbuf = PyTuple_GetItem(args,1);
    if( pynbuf == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: missing parameter");
        Py_RETURN_NONE;
    }
    PyObject* pybmem = PyTuple_GetItem(args,2);
    bool bdevmem = (bool)PyObject_IsTrue(pybmem);

    if( PyObject_CheckBuffer(pynbuf) )
    {
        Py_buffer nbuf;
        PyObject_GetBuffer(pynbuf,&nbuf,PyBUF_SIMPLE);
        unsigned char* nbuffer = (unsigned char*)nbuf.buf;
        Py_BEGIN_ALLOW_THREADS
        tptr->set_null_bitarray(nbuffer,false,bdevmem);
        Py_END_ALLOW_THREADS
        PyBuffer_Release(&nbuf);
    }
    else
    {
        unsigned char* nbuffer = (unsigned char*)PyLong_AsVoidPtr(pynbuf);
        Py_BEGIN_ALLOW_THREADS
        tptr->set_null_bitarray(nbuffer,false,bdevmem);
        Py_END_ALLOW_THREADS
    }
    Py_RETURN_NONE;
}

// compare a string to the list of strings
// a future method could compare a list to another list
static PyObject* n_compare( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->compare(str,devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    //
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->compare(str,rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

//
static PyObject* n_hash( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->hash(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int* rtn = new unsigned int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->hash(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

// convert the strings to integers
static PyObject* n_stoi( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->stoi(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->stoi(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

// convert the strings to long integers
static PyObject* n_stol( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    long* devptr = (long*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->stol(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    long* rtn = new long[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->stol(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyLong_FromLong(rtn[idx]));
    }
    delete rtn;
    return ret;
}

// convert the strings to floats
static PyObject* n_stof( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    float* devptr = (float*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->stof(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    float* rtn = new float[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->stof(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyFloat_FromDouble((double)rtn[idx]));
    }
    delete rtn;
    return ret;
}

// convert the strings to doubles
static PyObject* n_stod( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    double* devptr = (double*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->stod(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    double* rtn = new double[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->stod(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyFloat_FromDouble(rtn[idx]));
    }
    delete rtn;
    return ret;
}

// convert the strings with hex characters to integers
static PyObject* n_htoi( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->htoi(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int* rtn = new unsigned int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->htoi(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

// convert the strings with ip address to integers
static PyObject* n_ip2int( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->ip2int(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int* rtn = new unsigned int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->ip2int(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

// convert the strings with timestamps to integers
static PyObject* n_timestamp2int( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    PyObject* argFormat = PyTuple_GetItem(args,1);
    const char* format = 0;
    if( argFormat != Py_None )
        format = PyUnicode_AsUTF8(argFormat);
    PyObject* argUnits = PyTuple_GetItem(args,2);
    const char* unitsz = PyUnicode_AsUTF8(argUnits);
    std::string str_units = unitsz;
    if( name_units.find(str_units)==name_units.end() )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings: units parameter value unrecognized");
        Py_RETURN_NONE;
    }
    NVStrings::timestamp_units units = name_units[str_units];
    unsigned long* devptr = (unsigned long*)PyLong_AsVoidPtr(PyTuple_GetItem(args,3));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->timestamp2long(format,units,devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned long* rtn = new unsigned long[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->timestamp2long(format,units,rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyLong_FromLong(rtn[idx]));
    }
    delete rtn;
    return ret;
}

// convert the strings to booleans
static PyObject* n_to_bools( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    //
    PyObject* argTrue = PyTuple_GetItem(args,1);
    const char* tstr = 0;
    if( argTrue != Py_None )
        tstr = PyUnicode_AsUTF8(argTrue);

    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->to_bools(devptr,tstr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->to_bools(rtn,tstr,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        if( rtn[idx] )
        {
            Py_INCREF(Py_True);
            PyList_SetItem(ret, idx, Py_True);
        }
        else
        {
            Py_INCREF(Py_False);
            PyList_SetItem(ret, idx, Py_False);
        }
    }
    delete rtn;
    return ret;
}

// concatenate the given string to the end of all the strings
static PyObject* n_cat( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* argOthers = PyTuple_GetItem(args,1);
    PyObject* argSep = PyTuple_GetItem(args,2);
    PyObject* argNaRep = PyTuple_GetItem(args,3);

    const char* sep = "";
    if( argSep != Py_None )
        sep = PyUnicode_AsUTF8(argSep);

    const char* narep = 0;
    if( argNaRep != Py_None )
        narep = PyUnicode_AsUTF8(argNaRep);

    if( argOthers == Py_None )
    {
        // this is just a join -- need to account for the other parms too
        NVStrings* rtn = nullptr;
        Py_BEGIN_ALLOW_THREADS
        rtn = tptr->join(sep,narep);
        Py_END_ALLOW_THREADS
        if( rtn )
            return PyLong_FromVoidPtr((void*)rtn);
        Py_RETURN_NONE;
    }

    NVStrings* rtn = 0;
    std::string cname = argOthers->ob_type->tp_name;
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(argOthers);
        if( count==0 )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.cat empty argument list");
            Py_RETURN_NONE;
        }

        std::vector<NVStrings*> others;
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pystr = PyList_GetItem(argOthers,idx);
            if( pystr == Py_None )
            {
                PyErr_Format(PyExc_ValueError,"others list must not contain None");
                Py_RETURN_NONE;
            }
            std::string cname = pystr->ob_type->tp_name;
            if( cname.compare("nvstrings")!=0 )
            {
                PyErr_Format(PyExc_ValueError,"others list must contain nvstrings objects");
                Py_RETURN_NONE;
            }
            PyObject* pycptr = PyObject_GetAttrString(pystr,"m_cptr");
            NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(pycptr);
            others.push_back(strs);
        }
        Py_BEGIN_ALLOW_THREADS
        rtn = tptr->cat(others,sep,narep);
        Py_END_ALLOW_THREADS
    }
    else if( cname.compare("nvstrings")==0 )
    {
        NVStrings* others = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(argOthers,"m_cptr"));
        if( !others )
        {
            PyErr_Format(PyExc_ValueError,"invalid parameter");
            Py_RETURN_NONE;
        }
        //printf("others count=%d\n",others->size());
        if( others->size() != tptr->size() ) //Consider releasing the GIL here?
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.cat list size must match");
            Py_RETURN_NONE;
        }
        Py_BEGIN_ALLOW_THREADS
        rtn = tptr->cat(others,sep,narep);
        Py_END_ALLOW_THREADS
    }

    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// split each string into newer strings
// this will return an array of NVStrings to be wrapped in nvstrings
static PyObject* n_split_record( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delimiter = PyUnicode_AsUTF8(argOpt);
    int maxsplit = -1;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        maxsplit = (int)PyLong_AsLong(argOpt);

    std::vector<NVStrings*> results;
    Py_BEGIN_ALLOW_THREADS
    tptr->split_record(delimiter,maxsplit,results);
    Py_END_ALLOW_THREADS
    //
    PyObject* ret = PyList_New(tptr->size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// another split but from the right
static PyObject* n_rsplit_record( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delimiter = PyUnicode_AsUTF8(argOpt);
    int maxsplit = -1;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        maxsplit = (int)PyLong_AsLong(argOpt);

    std::vector<NVStrings*> results;
    Py_BEGIN_ALLOW_THREADS
    tptr->rsplit_record(delimiter,maxsplit,results);
    Py_END_ALLOW_THREADS
    //
    PyObject* ret = PyList_New(tptr->size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret, idx, PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

//
static PyObject* n_partition( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    //
    std::vector<NVStrings*> results;
    Py_BEGIN_ALLOW_THREADS
    tptr->partition(delimiter,results);
    Py_END_ALLOW_THREADS
    //
    PyObject* ret = PyList_New(tptr->size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

//
static PyObject* n_rpartition( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    //
    std::vector<NVStrings*> results;
    Py_BEGIN_ALLOW_THREADS
    tptr->rpartition(delimiter,results);
    Py_END_ALLOW_THREADS
    //
    PyObject* ret = PyList_New(tptr->size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

static PyObject* n_split( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delimiter = PyUnicode_AsUTF8(argOpt);
    int maxsplit = -1;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        maxsplit = (int)PyLong_AsLong(argOpt);

    std::vector<NVStrings*> results;
    int columns = 0;
    Py_BEGIN_ALLOW_THREADS
    columns = (int)tptr->split(delimiter,maxsplit,results);
    Py_END_ALLOW_THREADS
    //
    PyObject* ret = PyList_New(columns);
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

static PyObject* n_rsplit( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delimiter = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delimiter = PyUnicode_AsUTF8(argOpt);
    int maxsplit = -1;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        maxsplit = (int)PyLong_AsLong(argOpt);

    std::vector<NVStrings*> results;
    int columns = 0;
    Py_BEGIN_ALLOW_THREADS
    columns = (int)tptr->rsplit(delimiter,maxsplit,results);
    Py_END_ALLOW_THREADS
    //
    PyObject* ret = PyList_New(columns);
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// return a single character
// this will return a new NVStrings array where the strings are all single characters
static PyObject* n_get( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int position = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,1));

    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->get(position);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// repeat each string a number of times
static PyObject* n_repeat( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;
    unsigned int count = 0;
    if( !parse_args("repeat",args,"OI",&vo,&count) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->repeat(count);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// add padding around strings to a fixed size
static PyObject* n_pad( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;
    unsigned int width = 0;
    const char* side = 0;
    const char* fillchar = 0;
    if( !parse_args("pad",args,"OIzz",&vo,&width,&side,&fillchar) )
        Py_RETURN_NONE;
    if( *fillchar==0 )
    {
        PyErr_Format(PyExc_ValueError,"fillchar cannot be empty");
        Py_RETURN_NONE;
    }

    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings::padside ps = NVStrings::left;
    std::string sside = side;
    if( sside.compare("right")==0 )
        ps = NVStrings::right;
    else if( sside.compare("both")==0 )
        ps = NVStrings::both;

    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->pad(width,ps,fillchar);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// left-justify (and right pad) each string
static PyObject* n_ljust( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;
    unsigned int width = 0;
    const char* fillchar = 0;
    if( !parse_args("ljust",args,"OIz",&vo,&width,&fillchar) )
        Py_RETURN_NONE;
    if( *fillchar==0 )
    {
        PyErr_Format(PyExc_ValueError,"fillchar cannot be empty");
        Py_RETURN_NONE;
    }

    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->ljust(width,fillchar);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// center each string and pad right/left
static PyObject* n_center( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;
    unsigned int width = 0;
    const char* fillchar = 0;
    if( !parse_args("ljust",args,"OIz",&vo,&width,&fillchar) )
        Py_RETURN_NONE;
    if( *fillchar==0 )
    {
        PyErr_Format(PyExc_ValueError,"fillchar cannot be empty");
        Py_RETURN_NONE;
    }

    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->center(width,fillchar);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// right justify each string (and left pad)
static PyObject* n_rjust( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;
    unsigned int width = 0;
    const char* fillchar = 0;
    if( !parse_args("ljust",args,"OIz",&vo,&width,&fillchar) )
        Py_RETURN_NONE;
    if( *fillchar==0 )
    {
        PyErr_Format(PyExc_ValueError,"fillchar cannot be empty");
        Py_RETURN_NONE;
    }

    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->rjust(width,fillchar);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// zero pads strings correctly that contain numbers
static PyObject* n_zfill( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;
    unsigned int width = 0;
    if( !parse_args("zfill",args,"OI",&vo,&width) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->zfill(width);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// this attempts to do some kind of line wrapping by inserting new line characters
static PyObject* n_wrap( PyObject* self, PyObject* args )
{
    unsigned int width = 0;
    PyObject* vo = 0;
    if( !parse_args("wrap",args,"OI",&vo,&width) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->wrap(width);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// returns substring of each string
static PyObject* n_slice( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int start = PyLong_AsLong(PyTuple_GetItem(args,1));
    int end = -1, step = 1;
    PyObject* argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        end = (int)PyLong_AsLong(argOpt);
    argOpt = PyTuple_GetItem(args,3);
    if( argOpt != Py_None )
        step = (int)PyLong_AsLong(argOpt);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->slice(start,end,step);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// returns substring of each string using individual position values
static PyObject* n_slice_from( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int* starts = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    int* ends = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->slice_from(starts,ends);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// replaces the given range with the given string
static PyObject* n_slice_replace( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int start = 0, end = -1;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        start = (int)PyLong_AsLong(argOpt);
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        end = (int)PyLong_AsLong(argOpt);
    const char* repl = 0;
    argOpt = PyTuple_GetItem(args,3);
    if( argOpt != Py_None )
        repl = PyUnicode_AsUTF8(argOpt);
    //
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->slice_replace(repl,start,end);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// replace the string specified (if found) with the target string
static PyObject* n_replace( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;      // self pointer   = O
    const char* pat = 0;   // cannot be null = s
    const char* repl = 0;  // can be null    = z
    int maxrepl = -1;      // integer        = i
    int bregex = true;     // boolean        = p (do not use bool type here)
    if( !parse_args("replace",args,"Oszip",&vo,&pat,&repl,&maxrepl,&bregex) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = 0;
    std::string message;
    Py_BEGIN_ALLOW_THREADS
    try
    {
        if( bregex )
            rtn = tptr->replace_re(pat,repl,(int)maxrepl);
        else
            rtn = tptr->replace(pat,repl,(int)maxrepl);
    }
    catch(const std::exception& e)
    {
        message = e.what();
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_replace_multi( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* argPats = PyTuple_GetItem(args,1);
    PyObject* argRepls = PyTuple_GetItem(args,2);
    bool bregex = (bool)PyObject_IsTrue(PyTuple_GetItem(args,3));
    NVStrings* repls = (NVStrings*)PyLong_AsVoidPtr(argRepls);

    NVStrings* rtn = 0;
    std::string message;
    if( bregex )
    {
        // convert list into vector
        unsigned int count = (unsigned int)PyList_Size(argPats);
        std::vector<const char*> pats;
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pystr = PyList_GetItem(argPats,idx);
            if( pystr != Py_None )
                pats.push_back(PyUnicode_AsUTF8(pystr));
        }
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = tptr->replace_re(pats,*repls);
        }
        catch(const std::exception& e)
        {
            message = e.what();
        }
        Py_END_ALLOW_THREADS
    }
    else
    {
        NVStrings* ptns = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(argPats,"m_cptr"));
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = tptr->replace(*ptns,*repls);
        }
        catch(const std::exception& e)
        {
            message = e.what();
        }
        Py_END_ALLOW_THREADS
    }
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyObject* n_replace_with_backrefs( PyObject* self, PyObject* args )
{
    PyObject* vo = 0;      // self pointer   = O
    const char* pat = 0;   // cannot be null = s
    const char* repl = 0;  // can be null    = z
    if( !parse_args("replace_with_backrefs",args,"Osz",&vo,&pat,&repl) )
        Py_RETURN_NONE;
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(vo);
    NVStrings* rtn = 0;
    std::string message;
    Py_BEGIN_ALLOW_THREADS
    try
    {
        rtn = tptr->replace_with_backrefs(pat,repl);
    }
    catch(const std::exception& e)
    {
        message = e.what();
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_fillna( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    //if( !parse_args("fillna",args,"Os",&vo,&repl) )
    //    Py_RETURN_NONE;
    PyObject* pyrepl = PyTuple_GetItem(args,1);
    if( pyrepl == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.fillna repl argument must be specified");
        Py_RETURN_NONE;
    }
    NVStrings* rtn = nullptr;
    std::string cname = pyrepl->ob_type->tp_name;
    if( cname.compare("nvstrings")==0 )
    {
        NVStrings* trepl = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pyrepl,"m_cptr"));
        if( trepl->size() != tptr->size() )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.fillna repl argument must be same size");
            Py_RETURN_NONE;
        }
        Py_BEGIN_ALLOW_THREADS
        rtn = tptr->fillna(*trepl);
        Py_END_ALLOW_THREADS
    }
    else if( cname.compare("str")==0 )
    {
        const char* repl = PyUnicode_AsUTF8(pyrepl);
        Py_BEGIN_ALLOW_THREADS
        rtn = tptr->fillna(repl);
        Py_END_ALLOW_THREADS
    }
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// inserts a string into each string
static PyObject* n_insert( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    int start = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        start = (int)PyLong_AsLong(argOpt);
    const char* repl = 0;
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        repl = PyUnicode_AsUTF8(argOpt);
    //
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->insert(repl,start);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// strip specific characters from the beginning of each string
static PyObject* n_lstrip( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* to_strip = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        to_strip = PyUnicode_AsUTF8(argOpt);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->lstrip(to_strip);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// strip characters from the beginning and the end of each string
static PyObject* n_strip( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* to_strip = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        to_strip = PyUnicode_AsUTF8(argOpt);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->strip(to_strip);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// right strip characters from the end of each string
static PyObject* n_rstrip( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* to_strip = 0;
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        to_strip = PyUnicode_AsUTF8(argOpt);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->rstrip(to_strip);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// lowercase each string in place
static PyObject* n_lower( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->lower();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// uppercase each string in place
static PyObject* n_upper( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->upper();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// capitalize the first character of each string
static PyObject* n_capitalize( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->capitalize();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// swap the upper/lower case of each string's characters
static PyObject* n_swapcase( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->swapcase();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// title-case each string
static PyObject* n_title( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->title();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// search for the given string and return the positions it was found in each string
// https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.find.html
static PyObject* n_find( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int start = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    int end = -1;
    PyObject* argEnd = PyTuple_GetItem(args,3);
    if( argEnd != Py_None )
        end = (int)PyLong_AsLong(argEnd);
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->find(str,start,end,devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->find(str,start,end,rtn,false);
    Py_END_ALLOW_THREADS
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

// this was created out of a need to search for the 2nd occurrence of a string
static PyObject* n_find_from( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int* starts = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    int* ends = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,3));

    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->find_from(str,starts,ends,devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->find_from(str,starts,ends,rtn,false);
    Py_END_ALLOW_THREADS
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

// right-search for the given string and return the positions it was found in each string
static PyObject* n_rfind( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int start = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    int end = -1;
    PyObject* argEnd = PyTuple_GetItem(args,3);
    if( argEnd != Py_None )
        end = (int)PyLong_AsLong(argEnd);
    //
    unsigned int count = tptr->size();
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->rfind(str,start,end,devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->rfind(str,start,end,rtn,false);
    Py_END_ALLOW_THREADS
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
    }
    delete rtn;
    return ret;
}

// return position of each string provided
static PyObject* n_find_multiple( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* argStrs = PyTuple_GetItem(args,1);
    if( argStrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.find_multiple strs argument must be specified");
        Py_RETURN_NONE;
    }
    NVStrings* strs = 0;
    std::string cname = argStrs->ob_type->tp_name;
    if( cname.compare("nvstrings")==0 )
        strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(argStrs,"m_cptr"));
    else if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(argStrs);
        if( count )
        {
            const char** list = new const char*[count];
            for( unsigned int idx=0; idx < count; ++idx )
            {
                PyObject* pystr = PyList_GetItem(argStrs,idx);
                if( (pystr == Py_None) || !PyObject_TypeCheck(pystr,&PyUnicode_Type) )
                    list[idx] = 0;
                else
                    list[idx] = PyUnicode_AsUTF8(pystr);
            }
            Py_BEGIN_ALLOW_THREADS
            strs = NVStrings::create_from_array(list,count);
            Py_END_ALLOW_THREADS
            delete list;
        }
    }
    //
    if( !strs )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.find_multiple invalid strs parameter");
        Py_RETURN_NONE;
    }
    if( strs->size()==0 )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.find_multiple empty strs list");
        Py_RETURN_NONE;
    }

    // resolve output pointer
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->find_multiple(*strs,devptr);
        Py_END_ALLOW_THREADS
        if( cname.compare("list")==0 )
        {
            Py_BEGIN_ALLOW_THREADS
            NVStrings::destroy(strs); // destroy it if we made it (above)
            Py_END_ALLOW_THREADS
        }
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int rows = tptr->size();
    PyObject* ret = PyList_New(rows);
    if( rows==0 )
    {
        if( cname.compare("list")==0 )
        {
            Py_BEGIN_ALLOW_THREADS
            NVStrings::destroy(strs); // destroy it if we made it (above)
            Py_END_ALLOW_THREADS
        }
        return ret;
    }
    //
    unsigned int columns = strs->size();
    int* rtn = new int[rows*columns];
    Py_BEGIN_ALLOW_THREADS
    tptr->find_multiple(*strs,rtn,false);
    Py_END_ALLOW_THREADS
    for(unsigned int idx=0; idx < rows; ++idx)
    {
        PyObject* row = PyList_New(columns);
        for( unsigned int jdx=0; jdx < columns; ++jdx )
        {
            int val = rtn[(idx*columns)+jdx];
            if( val < -1 )
            {
                Py_INCREF(Py_None);
                PyList_SetItem(row, jdx, Py_None);
            }
            else
                PyList_SetItem(row, jdx, PyLong_FromLong((long)val));
        }
        PyList_SetItem(ret, idx, row);
    }
    delete rtn;
    //
    if( cname.compare("list")==0 )
    {
        Py_BEGIN_ALLOW_THREADS
        NVStrings::destroy(strs); // destroy it if we made it (above)
        Py_END_ALLOW_THREADS
    }
    return ret;
}

// this is the same as find but throws an error if string is not found
// https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.index.html
static PyObject* n_index( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int start = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    int end = -1;
    PyObject* argEnd = PyTuple_GetItem(args,3);
    if( argEnd != Py_None )
        end = (int)PyLong_AsLong(argEnd);
    //
    unsigned int count = tptr->size();
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        unsigned int success = 0;
        Py_BEGIN_ALLOW_THREADS
        success = tptr->find(str,start,end,devptr);
        Py_END_ALLOW_THREADS
        if( success != count )
            PyErr_Format(PyExc_ValueError,"nvstrings.index: [%s] not found in %d elements",str,(int)(count-success));
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->find(str,start,end,rtn,false);
    Py_END_ALLOW_THREADS
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else if( val >= 0 )
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
        else
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.index: [%s] not found in element %d",str,(int)idx);
            break;
        }
    }
    delete rtn;
    return ret;
}

// same as rfind excepts throws an error if string is not found
static PyObject* n_rindex( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int start = (int)PyLong_AsLong(PyTuple_GetItem(args,2));
    int end = -1;
    PyObject* argEnd = PyTuple_GetItem(args,3);
    if( argEnd != Py_None )
        end = (int)PyLong_AsLong(argEnd);
    //
    unsigned int count = tptr->size();
    //
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        size_t success = 0;
        Py_BEGIN_ALLOW_THREADS
        success = tptr->rfind(str,start,end,devptr);
        Py_END_ALLOW_THREADS
        if( success != count )
            PyErr_Format(PyExc_ValueError,"nvstrings.rindex: [%s] not found in %d elements",str,(int)(count-success));
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    int* rtn = new int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->rfind(str,start,end,rtn,false);
    Py_END_ALLOW_THREADS
    for(unsigned int idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val < -1 )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
        else if( val >= 0 )
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
        else
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.rindex: [%s] not found in element %d",str,(int)idx);
            break;
        }
    }
    delete rtn;
    return ret;
}

// this will return an array of NVStrings to be wrapped in nvstrings
static PyObject* n_findall_record( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* pat = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));

    std::string message;
    std::vector<NVStrings*> results;
    Py_BEGIN_ALLOW_THREADS
    try
    {
        tptr->findall_record(pat,results);
    }
    catch(const std::exception& e)
    {
        message = e.what();
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    //
    PyObject* ret = PyList_New(results.size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// same but column-major groupings of results
static PyObject* n_findall( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* pat = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));

    std::string message;
    std::vector<NVStrings*> results;
    Py_BEGIN_ALLOW_THREADS
    try
    {
        tptr->findall(pat,results);
    }
    catch(const std::exception& e)
    {
        message = e.what();
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    //
    PyObject* ret = PyList_New(results.size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}


// This can take a regex string too
// https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.contains.html
static PyObject* n_contains( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    //
    bool bregex = (bool)PyObject_IsTrue(PyTuple_GetItem(args,2));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,3));
    int rc = 0;
    std::string message;
    if( devptr )
    {
        //Save thread state and release the GIL as we do not operate on PyObjects
        Py_BEGIN_ALLOW_THREADS
        try
        {
            if( bregex )
                rc = tptr->contains_re(str,devptr);
            else
                rc = tptr->contains(str,devptr);
        }
        catch(const std::exception& e)
        {
            message = e.what();
            rc = -1;
        }
        //Restore thread state and acquire the GIL again.
        Py_END_ALLOW_THREADS
        if( !message.empty() )
            PyErr_Format(PyExc_ValueError,message.c_str());
        if( rc < 0 )
            Py_RETURN_NONE;
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);
    bool* rtn = new bool[count];

    Py_BEGIN_ALLOW_THREADS
    try
    {
        if( bregex )
            rc = tptr->contains_re(str,rtn,false);
        else
            rc = tptr->contains(str,rtn,false);
    }
    catch(const std::exception& e)
    {
        message = e.what();
        rc = -1;
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    if( rc < 0 )
    {
        delete rtn;
        Py_RETURN_NONE;
    }
    PyObject* ret = PyList_New(count);
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_match( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    //
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    int rc = 0;
    std::string message;
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rc = tptr->match(str,devptr);
        }
        catch(const std::exception& e)
        {
            message = e.what();
            rc = -1;
        }
        Py_END_ALLOW_THREADS
        if( !message.empty() )
            PyErr_Format(PyExc_ValueError,message.c_str());
        if( rc < 0 )
            Py_RETURN_NONE;
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    try
    {
        rc = tptr->match(str,rtn,false);
    }
    catch(const std::exception& e)
    {
        message = e.what();
        rc = -1;
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    if( rc < 0 )
    {
        delete rtn;
        Py_RETURN_NONE;
    }
    PyObject* ret = PyList_New(count);
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_match_strings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pystrs = PyTuple_GetItem(args,1);
    if( pystrs == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.match_strings: parameter required");
        Py_RETURN_NONE;
    }
    NVStrings* strs = 0;
    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pystrs);
        if( count==0 )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.match_strings empty argument list");
            Py_RETURN_NONE;
        }
        if( count != (int)tptr->size() )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.match_strings list size must match");
            Py_RETURN_NONE;
        }
        const char** list = new const char*[count];
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pystr = PyList_GetItem(pystrs,idx);
            if( (pystr == Py_None) || !PyObject_TypeCheck(pystr,&PyUnicode_Type) )
                list[idx] = 0;
            else
                list[idx] = PyUnicode_AsUTF8(pystr);
        }
        Py_BEGIN_ALLOW_THREADS
        strs = NVStrings::create_from_array(list,count);
        Py_END_ALLOW_THREADS
        delete list;
    }
    // or a single nvstrings instance
    else if( cname.compare("nvstrings")==0 )
    {
        strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
        if( strs==0 )
        {
            PyErr_Format(PyExc_ValueError,"nvstrings.match_strings: invalid nvstrings object");
            Py_RETURN_NONE;
        }
    }
    else
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.match_strings: argument must be nvstrings object");
        Py_RETURN_NONE;
    }

    //
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    int rc = 0;
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        rc = tptr->match_strings(*strs,devptr);
        Py_END_ALLOW_THREADS
        if( cname.compare("list")==0 )
        {
            Py_BEGIN_ALLOW_THREADS
            NVStrings::destroy(strs); // destroy it if we made it (above)
            Py_END_ALLOW_THREADS
        }
        if( rc < 0 )
            Py_RETURN_NONE;
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    rc = tptr->match_strings(*strs,rtn,false);
    Py_END_ALLOW_THREADS
    if( cname.compare("list")==0 )
    {
        Py_BEGIN_ALLOW_THREADS
        NVStrings::destroy(strs); // destroy it if we made it (above)
        Py_END_ALLOW_THREADS
    }
    if( rc < 0 )
    {
        delete rtn;
        Py_RETURN_NONE;
    }
    PyObject* ret = PyList_New(count);
    for(size_t idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

//
static PyObject* n_startswith( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->startswith(str,devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->startswith(str,rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

//
static PyObject* n_endswith( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->endswith(str,devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->endswith(str,rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

//
static PyObject* n_count( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    int* devptr = (int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,2));
    std::string message;
    if( devptr )
    {
        int rc  = 0;
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rc = tptr->count_re(str,devptr);
        }
        catch(const std::exception& e)
        {
            message = e.what();
            rc = -1;
        }
        Py_END_ALLOW_THREADS
        if( !message.empty() )
            PyErr_Format(PyExc_ValueError,message.c_str());
        if( rc < 0 )
            Py_RETURN_NONE;
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    if( count==0 )
        return PyList_New(0);
    int* rtn = new int[count];
    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    try
    {
        rc = tptr->count_re(str,rtn,false);
    }
    catch(const std::exception& e)
    {
        message = e.what();
        rc = -1;
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    if( rc < 0 )
    {
        delete rtn;
        Py_RETURN_NONE;
    }
    //
    PyObject* ret = PyList_New(count);
    for(size_t idx=0; idx < count; idx++)
    {
        int val = rtn[idx];
        if( val >= 0 )
            PyList_SetItem(ret, idx, PyLong_FromLong((long)val));
        else
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
        }
    }
    delete rtn;
    return ret;
}

// this will return an array of NVStrings to be wrapped in nvstrings
static PyObject* n_extract_record( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* pat = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));

    std::string message;
    std::vector<NVStrings*> results;
    Py_BEGIN_ALLOW_THREADS
    try
    {
        tptr->extract_record(pat,results);
    }
    catch(const std::exception& e)
    {
        message = e.what();
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    //
    PyObject* ret = PyList_New(results.size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// same but column-major groupings of results
static PyObject* n_extract( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* pat = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));

    std::string message;
    std::vector<NVStrings*> results;
    Py_BEGIN_ALLOW_THREADS
    try
    {
        tptr->extract(pat,results);
    }
    catch(const std::exception& e)
    {
        message = e.what();
    }
    Py_END_ALLOW_THREADS
    if( !message.empty() )
        PyErr_Format(PyExc_ValueError,message.c_str());
    //
    PyObject* ret = PyList_New(results.size());
    int idx=0;
    for( auto itr=results.begin(); itr != results.end(); itr++,idx++ )
        PyList_SetItem(ret,idx,PyLong_FromVoidPtr((void*)*itr));
    return ret;
}

// This translates each character based on a given table.
// The table can be an array of pairs--array with 2 values or a dictionary created by str.maketrans()
static PyObject* n_translate( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pytable = PyTuple_GetItem(args,1);
    std::string cname = pytable->ob_type->tp_name; // list or dict

    unsigned int count = 0;
    std::vector< std::pair<unsigned,unsigned> > table;
    if( cname.compare("list")==0 )
    {
        // convert table parm into pair<unsigned,unsigned> array
        count = (unsigned int)PyList_Size(pytable);
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* pyentry = PyList_GetItem(pytable,idx);
            if( PyList_Size(pyentry)!=2 )
            {
                PyErr_Format(PyExc_ValueError,"nvstrings.translate: invalid map entry");
                Py_RETURN_NONE;
            }
            std::pair<unsigned,unsigned> entry;
            PyUnicode_AsWideChar(PyList_GetItem(pyentry,0),(wchar_t*)&(entry.first),1);
            PyObject* pysecond = PyList_GetItem(pyentry,1);
            if( pysecond != Py_None )
                PyUnicode_AsWideChar(pysecond,(wchar_t*)&(entry.second),1);
            else
                entry.second = 0;
            table.push_back(entry);
        }
    }
    else if( cname.compare("dict")==0 )
    {
        count = (unsigned int)PyDict_Size(pytable);
        PyObject* items = PyDict_Items(pytable);
        for( unsigned int idx=0; idx < count; ++idx )
        {
            PyObject* item = PyList_GetItem(items,idx);
            std::pair<unsigned,unsigned> entry;
            entry.first = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(item,0));
            PyObject* item1 = PyTuple_GetItem(item,1);
            if( item1 != Py_None )
                entry.second = (unsigned)PyLong_AsUnsignedLong(item1);
            else
                entry.second = 0;
            table.push_back(entry);
        }
    }
    else
    {
        PyErr_Format(PyExc_ValueError,"nvstrings.translate: invalid argument type");
        Py_RETURN_NONE;
    }
    //
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->translate(table.data(),count);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// joins two strings with these strings in the middle
static PyObject* n_join( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* delim = "";
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        delim = PyUnicode_AsUTF8(argOpt);
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->join(delim);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// sorts the strings by length/name
static PyObject* n_sort( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings::sorttype stype = (NVStrings::sorttype)PyLong_AsLong(PyTuple_GetItem(args,1));
    bool asc = (bool)PyObject_IsTrue(PyTuple_GetItem(args,2));
    bool nullfirst = (bool)PyObject_IsTrue(PyTuple_GetItem(args,3));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->sort(stype,asc,nullfirst);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

// like sort but returns new index order only
static PyObject* n_order( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings::sorttype stype = (NVStrings::sorttype)PyLong_AsLong(PyTuple_GetItem(args,1));
    bool asc = (bool)PyObject_IsTrue(PyTuple_GetItem(args,2));
    bool nullfirst = (bool)PyObject_IsTrue(PyTuple_GetItem(args,3));
    unsigned int* devptr = (unsigned int*)PyLong_AsVoidPtr(PyTuple_GetItem(args,4));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->order(stype,asc,devptr,nullfirst);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }

    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    unsigned int* rtn = new unsigned int[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->order(stype,asc,rtn,nullfirst,false);
    Py_END_ALLOW_THREADS
    for(unsigned int idx=0; idx < count; idx++)
        PyList_SetItem(ret, idx, PyLong_FromLong((long)rtn[idx]));
    delete rtn;
    return ret;
}

//
static PyObject* n_gather( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyidxs = PyTuple_GetItem(args,1);

    DataBuffer<int> dbvalues(pyidxs);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.n_gather(): %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( !dbvalues.is_blist() && dbvalues.get_type_width()!=sizeof(int) )
    {
        PyErr_Format(PyExc_TypeError,"nvstrings.n_gather(): values must be of type int32");
        Py_RETURN_NONE;
    }

    unsigned int count = dbvalues.get_count();
    if( count==0 )
        count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,2));
    bool bdevmem = dbvalues.is_device_type();

    NVStrings* rtn = 0;
    std::string message;
    if( !dbvalues.is_blist() )
    {
        Py_BEGIN_ALLOW_THREADS
        try
        {
            rtn = tptr->gather(dbvalues.get_values(),count,bdevmem);
        }
        catch(const std::out_of_range& eor)
        {
            std::ostringstream errmsg;
            errmsg << "one or more indexes out of range [0:" << tptr->size() << ")";
            message = errmsg.str();
        }
        Py_END_ALLOW_THREADS
    }
    else
    {
        // handle boolean values too
        DataBuffer<bool> dbmask(pyidxs);
        Py_BEGIN_ALLOW_THREADS
        rtn = tptr->gather(dbmask.get_values(),bdevmem);
        Py_END_ALLOW_THREADS
    }

    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    if( !message.empty() )
        PyErr_Format(PyExc_IndexError,message.c_str());
    Py_RETURN_NONE;
}

//
static PyObject* n_sublist( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    unsigned int start = 0, end = tptr->size();
    PyObject* argOpt = PyTuple_GetItem(args,1);
    if( argOpt != Py_None )
        start = (unsigned int)PyLong_AsLong(argOpt);
    argOpt = PyTuple_GetItem(args,2);
    if( argOpt != Py_None )
        end = (unsigned int)PyLong_AsLong(argOpt);
    argOpt = PyTuple_GetItem(args,3);
    int step = 1;
    if( argOpt != Py_None )
        step = (int)PyLong_AsLong(argOpt);
    //
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->sublist(start,end,step);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyObject* n_scatter( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pystrs = PyTuple_GetItem(args,1);

    std::string cname = pystrs->ob_type->tp_name;
    if( cname.compare("nvstrings")!=0 )
    {
        PyErr_Format(PyExc_TypeError,"scatter: strs must be nvstrings type");
        Py_RETURN_NONE;
    }
    NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));

    PyObject* pyidxs = PyTuple_GetItem(args,2);
    DataBuffer<int> dbvalues(pyidxs);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"scatter: %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( dbvalues.get_type_width()!=sizeof(int) )
    {
        PyErr_Format(PyExc_TypeError,"scatter: values must be of type int32");
        Py_RETURN_NONE;
    }
    unsigned int count = dbvalues.get_count();
    if( count && (count < strs->size()) )
    {
        PyErr_Format(PyExc_ValueError,"scatter: number of values must match the number of strings in strs argument");
        Py_RETURN_NONE;
    }

    bool bdevmem = dbvalues.is_device_type();

    NVStrings* rtn = 0;
    std::string message;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->scatter(*strs,dbvalues.get_values(),bdevmem);
    Py_END_ALLOW_THREADS

    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    if( !message.empty() )
        PyErr_Format(PyExc_IndexError,message.c_str());
    Py_RETURN_NONE;
}

static PyObject* n_scalar_scatter( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    const char* repl_str = PyUnicode_AsUTF8(PyTuple_GetItem(args,1));
    PyObject* pyidxs = PyTuple_GetItem(args,2);
    DataBuffer<int> dbvalues(pyidxs);
    if( dbvalues.is_error() )
    {
        PyErr_Format(PyExc_TypeError,"scatter: %s",dbvalues.get_error_text());
        Py_RETURN_NONE;
    }
    if( dbvalues.get_type_width()!=sizeof(int) )
    {
        PyErr_Format(PyExc_TypeError,"scatter: values must be of type int32");
        Py_RETURN_NONE;
    }
    unsigned int count = dbvalues.get_count();
    if( !count )
        count = (unsigned int)PyLong_AsLong(PyTuple_GetItem(args,3));
    bool bdevmem = dbvalues.is_device_type();

    NVStrings* rtn = 0;
    std::string message;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->scatter(repl_str,dbvalues.get_values(),count,bdevmem);
    Py_END_ALLOW_THREADS

    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    if( !message.empty() )
        PyErr_Format(PyExc_IndexError,message.c_str());
    Py_RETURN_NONE;
}

//
static PyObject* n_remove_strings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyidxs = PyTuple_GetItem(args,1);
    std::string cname = pyidxs->ob_type->tp_name;
    NVStrings* rtn = 0;
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pyidxs);
        int* indexes = new int[count];
        for( int idx=0; idx < count; ++idx )
        {
            PyObject* pyidx = PyList_GetItem(pyidxs,idx);
            indexes[idx] = (int)PyLong_AsLong(pyidx);
        }
        Py_BEGIN_ALLOW_THREADS
        rtn = tptr->remove_strings(indexes,count,false);
        Py_END_ALLOW_THREADS
        delete indexes;
    }
    else // device pointer is expected
    {
        unsigned int count = 0;
        PyObject *vo=0, *dptr=0;
        if( !parse_args("remove_strings",args,"OOI",&vo,&dptr,&count) )
            Py_RETURN_NONE;
        const int* indexes = (const int*)PyLong_AsVoidPtr(dptr);
        Py_BEGIN_ALLOW_THREADS
        rtn = tptr->remove_strings(indexes,count);
        Py_END_ALLOW_THREADS
    }
    //
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

//
static PyObject* n_add_strings( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyarg = PyTuple_GetItem(args,1);
    std::string cname = pyarg->ob_type->tp_name;
    std::vector<NVStrings*> strslist;
    strslist.push_back(tptr);
    if( cname.compare("list")==0 )
    {
        unsigned int count = (unsigned int)PyList_Size(pyarg);
        for( int idx=0; idx < count; ++idx )
        {
            PyObject* pystrs = PyList_GetItem(pyarg,idx);
            NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pystrs,"m_cptr"));
            strslist.push_back(strs);
        }
    }
    else if( cname.compare("nvstrings")==0 )
    {
        NVStrings* strs = (NVStrings*)PyLong_AsVoidPtr(PyObject_GetAttrString(pyarg,"m_cptr"));
        strslist.push_back(strs);
    }
    //
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = NVStrings::create_from_strings(strslist);
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_copy( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->copy();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_isalnum( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->isalnum(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->isalnum(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_isalpha( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->isalpha(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->isalpha(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_isdigit( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->isdigit(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->isdigit(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_isspace( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->isspace(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->isspace(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_isdecimal( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->isdecimal(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->isdecimal(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_isnumeric( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->isnumeric(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->isnumeric(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_islower( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->islower(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->islower(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_isupper( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->isupper(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->isupper(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_is_empty( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    bool* devptr = (bool*)PyLong_AsVoidPtr(PyTuple_GetItem(args,1));
    if( devptr )
    {
        Py_BEGIN_ALLOW_THREADS
        tptr->is_empty(devptr);
        Py_END_ALLOW_THREADS
        return PyLong_FromVoidPtr((void*)devptr);
    }
    // copy to host option
    unsigned int count = tptr->size();
    PyObject* ret = PyList_New(count);
    if( count==0 )
        return ret;
    bool* rtn = new bool[count];
    Py_BEGIN_ALLOW_THREADS
    tptr->is_empty(rtn,false);
    Py_END_ALLOW_THREADS
    std::vector<unsigned char> nulls(((count+7)/8),0);
    unsigned int ncount = 0;
    Py_BEGIN_ALLOW_THREADS
    ncount = tptr->set_null_bitarray(nulls.data(),false,false);
    Py_END_ALLOW_THREADS
    for(size_t idx=0; idx < count; idx++)
    {
        if( ncount && ((nulls[idx/8] & (1 << (idx % 8)))==0) )
        {
            Py_INCREF(Py_None);
            PyList_SetItem(ret, idx, Py_None);
            continue;
        }
        PyList_SetItem(ret, idx, PyBool_FromLong((long)rtn[idx]));
    }
    delete rtn;
    return ret;
}

static PyObject* n_get_info( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));

    StringsStatistics stats;
    Py_BEGIN_ALLOW_THREADS
    tptr->compute_statistics(stats);
    Py_END_ALLOW_THREADS

    PyObject* pydict = PyDict_New();
    PyDict_SetItemString(pydict,"total_strings",PyLong_FromLong(stats.total_strings));
    PyDict_SetItemString(pydict,"null_strings",PyLong_FromLong(stats.total_nulls));
    PyDict_SetItemString(pydict,"empty_strings",PyLong_FromLong(stats.total_empty));
    PyDict_SetItemString(pydict,"unique_strings",PyLong_FromLong(stats.unique_strings));
    PyDict_SetItemString(pydict,"total_bytes",PyLong_FromLong(stats.total_bytes));
    PyDict_SetItemString(pydict,"total_chars",PyLong_FromLong(stats.total_chars));
    PyDict_SetItemString(pydict,"device_memory",PyLong_FromLong(stats.total_memory));
    PyDict_SetItemString(pydict,"bytes_avg",PyLong_FromLong(stats.bytes_avg));
    PyDict_SetItemString(pydict,"bytes_min",PyLong_FromLong(stats.bytes_min));
    PyDict_SetItemString(pydict,"bytes_max",PyLong_FromLong(stats.bytes_max));
    PyDict_SetItemString(pydict,"chars_avg",PyLong_FromLong(stats.chars_avg));
    PyDict_SetItemString(pydict,"chars_min",PyLong_FromLong(stats.chars_min));
    PyDict_SetItemString(pydict,"chars_max",PyLong_FromLong(stats.chars_max));
    PyDict_SetItemString(pydict,"memory_avg",PyLong_FromLong(stats.mem_avg));
    PyDict_SetItemString(pydict,"memory_min",PyLong_FromLong(stats.mem_min));
    PyDict_SetItemString(pydict,"memory_max",PyLong_FromLong(stats.mem_max));
    PyDict_SetItemString(pydict,"whitespace",PyLong_FromLong(stats.whitespace_count));
    PyDict_SetItemString(pydict,"digits",PyLong_FromLong(stats.digits_count));
    PyDict_SetItemString(pydict,"uppercase",PyLong_FromLong(stats.uppercase_count));
    PyDict_SetItemString(pydict,"lowercase",PyLong_FromLong(stats.lowercase_count));

    PyObject* pyhist = PyDict_New();
    size_t count = stats.char_counts.size();
    for( size_t idx=0; idx < count; ++idx )
    {
        unsigned int chr = stats.char_counts[idx].first;
        unsigned int num = stats.char_counts[idx].second;
        unsigned char out[5] = {0,0,0,0,0};
        unsigned char* ptr = out + ((chr & 0xF0000000)==0xF0000000) + ((chr & 0xFFE00000)==0x00E00000) + ((chr & 0xFFFFC000)==0x0000C000);
        //printf("%p,%p,%x,%d\n",out,ptr,(chr & 0xFFFF),(int)((chr & 0xFFFFC000)==0x0000C00000));
        unsigned int cvt = chr;
        while( cvt > 0 )
        {
            *ptr-- = (unsigned char)(cvt & 255);
            cvt = cvt >> 8;
        }
        PyDict_SetItemString(pyhist,(const char*)out,PyLong_FromLong(num));
        //printf("    [%s] 0x%04x = %u\n",out,chr,num);
    }

    PyDict_SetItemString(pydict,"chars_histogram",pyhist);
    return pydict;
}

static PyObject* n_url_encode( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->url_encode();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}

static PyObject* n_url_decode( PyObject* self, PyObject* args )
{
    NVStrings* tptr = (NVStrings*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    NVStrings* rtn = nullptr;
    Py_BEGIN_ALLOW_THREADS
    rtn = tptr->url_decode();
    Py_END_ALLOW_THREADS
    if( rtn )
        return PyLong_FromVoidPtr((void*)rtn);
    Py_RETURN_NONE;
}


// Version 0.1, 0.1.1, 0.2, 0.2.1 features
static PyMethodDef s_Methods[] = {
    { "n_createFromHostStrings", n_createFromHostStrings, METH_VARARGS, "" },
    { "n_destroyStrings", n_destroyStrings, METH_VARARGS, "" },
    { "n_createFromIPC", n_createFromIPC, METH_VARARGS, "" },
    { "n_getIPCData", n_getIPCData, METH_VARARGS, "" },
    { "n_createHostStrings", n_createHostStrings, METH_VARARGS, "" },
    { "n_createFromCSV", n_createFromCSV, METH_VARARGS, "" },
    { "n_createFromOffsets", n_createFromOffsets, METH_VARARGS, "" },
    { "n_createFromNVStrings", n_createFromNVStrings, METH_VARARGS, "" },
    { "n_createFromInt32s", n_createFromInt32s, METH_VARARGS, "" },
    { "n_createFromInt64s", n_createFromInt64s, METH_VARARGS, "" },
    { "n_createFromFloat32s", n_createFromFloat32s, METH_VARARGS, "" },
    { "n_createFromFloat64s", n_createFromFloat64s, METH_VARARGS, "" },
    { "n_createFromIPv4Integers", n_createFromIPv4Integers, METH_VARARGS, "" },
    { "n_createFromTimestamp", n_createFromTimestamp, METH_VARARGS, "" },
    { "n_createFromBools", n_createFromBools, METH_VARARGS, "" },
    { "n_create_offsets", n_create_offsets, METH_VARARGS, "" },
    { "n_size", n_size, METH_VARARGS, "" },
    { "n_hash", n_hash, METH_VARARGS, "" },
    { "n_set_null_bitmask", n_set_null_bitmask, METH_VARARGS, "" },
    { "n_null_count", n_null_count, METH_VARARGS, "" },
    { "n_copy", n_copy, METH_VARARGS, "" },
    { "n_remove_strings", n_remove_strings, METH_VARARGS, "" },
    { "n_add_strings", n_add_strings, METH_VARARGS, "" },
    { "n_compare", n_compare, METH_VARARGS, "" },
    { "n_stoi", n_stoi, METH_VARARGS, "" },
    { "n_stol", n_stol, METH_VARARGS, "" },
    { "n_stof", n_stof, METH_VARARGS, "" },
    { "n_stod", n_stod, METH_VARARGS, "" },
    { "n_htoi", n_htoi, METH_VARARGS, "" },
    { "n_ip2int", n_ip2int, METH_VARARGS, "" },
    { "n_timestamp2int", n_timestamp2int, METH_VARARGS, "" },
    { "n_to_bools", n_to_bools, METH_VARARGS, "" },
    { "n_cat", n_cat, METH_VARARGS, "" },
    { "n_split", n_split, METH_VARARGS, "" },
    { "n_rsplit", n_rsplit, METH_VARARGS, "" },
    { "n_partition", n_partition, METH_VARARGS, "" },
    { "n_rpartition", n_rpartition, METH_VARARGS, "" },
    { "n_split_record", n_split_record, METH_VARARGS, "" },
    { "n_rsplit_record", n_rsplit_record, METH_VARARGS, "" },
    { "n_get", n_get, METH_VARARGS, "" },
    { "n_repeat", n_repeat, METH_VARARGS, "" },
    { "n_pad", n_pad, METH_VARARGS, "" },
    { "n_ljust", n_ljust, METH_VARARGS, "" },
    { "n_center", n_center, METH_VARARGS, "" },
    { "n_rjust", n_rjust, METH_VARARGS, "" },
    { "n_wrap", n_wrap, METH_VARARGS, "" },
    { "n_slice", n_slice, METH_VARARGS, "" },
    { "n_slice_from", n_slice_from, METH_VARARGS, "" },
    { "n_slice_replace", n_slice_replace, METH_VARARGS, "" },
    { "n_replace", n_replace, METH_VARARGS, "" },
    { "n_replace_multi", n_replace_multi, METH_VARARGS, "" },
    { "n_replace_with_backrefs", n_replace_with_backrefs, METH_VARARGS, "" },
    { "n_fillna", n_fillna, METH_VARARGS, "" },
    { "n_insert", n_insert, METH_VARARGS, "" },
    { "n_len", n_len, METH_VARARGS, "" },
    { "n_byte_count", n_byte_count, METH_VARARGS, "" },
    { "n_lstrip", n_lstrip, METH_VARARGS, "" },
    { "n_strip", n_strip, METH_VARARGS, "" },
    { "n_rstrip", n_rstrip, METH_VARARGS, "" },
    { "n_lower", n_lower, METH_VARARGS, "" },
    { "n_upper", n_upper, METH_VARARGS, "" },
    { "n_capitalize", n_capitalize, METH_VARARGS, "" },
    { "n_swapcase", n_swapcase, METH_VARARGS, "" },
    { "n_title", n_title, METH_VARARGS, "" },
    { "n_translate", n_translate, METH_VARARGS, "" },
    { "n_join", n_join, METH_VARARGS, "" },
    { "n_zfill", n_zfill, METH_VARARGS, "" },
    { "n_find", n_find, METH_VARARGS, "" },
    { "n_find_from", n_find_from, METH_VARARGS, "" },
    { "n_find_multiple", n_find_multiple, METH_VARARGS, "" },
    { "n_rfind", n_rfind, METH_VARARGS, "" },
    { "n_index", n_index, METH_VARARGS, "" },
    { "n_rindex", n_rindex, METH_VARARGS, "" },
    { "n_rindex", n_rindex, METH_VARARGS, "" },
    { "n_findall", n_findall, METH_VARARGS, "" },
    { "n_findall_record", n_findall_record, METH_VARARGS, "" },
    { "n_contains", n_contains, METH_VARARGS, "" },
    { "n_match", n_match, METH_VARARGS, "" },
    { "n_match_strings", n_match_strings, METH_VARARGS, "" },
    { "n_count", n_count, METH_VARARGS, "" },
    { "n_extract", n_extract, METH_VARARGS, "" },
    { "n_extract_record", n_extract_record, METH_VARARGS, "" },
    { "n_startswith", n_startswith, METH_VARARGS, "" },
    { "n_endswith", n_endswith, METH_VARARGS, "" },
    { "n_sort", n_sort, METH_VARARGS, "" },
    { "n_order", n_order, METH_VARARGS, "" },
    { "n_gather", n_gather, METH_VARARGS, "" },
    { "n_sublist", n_sublist, METH_VARARGS, "" },
    { "n_scatter", n_scatter, METH_VARARGS, "" },
    { "n_scalar_scatter", n_scalar_scatter, METH_VARARGS, "" },
    { "n_isalnum", n_isalnum, METH_VARARGS, "" },
    { "n_isalpha", n_isalpha, METH_VARARGS, "" },
    { "n_isdigit", n_isdigit, METH_VARARGS, "" },
    { "n_isspace", n_isspace, METH_VARARGS, "" },
    { "n_isdecimal", n_isdecimal, METH_VARARGS, "" },
    { "n_isnumeric", n_isnumeric, METH_VARARGS, "" },
    { "n_islower", n_islower, METH_VARARGS, "" },
    { "n_isupper", n_isupper, METH_VARARGS, "" },
    { "n_is_empty", n_is_empty, METH_VARARGS, "" },
    { "n_get_info", n_get_info, METH_VARARGS, "" },
    { "n_url_encode", n_url_encode, METH_VARARGS, "" },
    { "n_url_decode", n_url_decode, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem = {	PyModuleDef_HEAD_INIT, "NVStrings_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_pyniNVStrings(void)
{
    //NVStrings::initLibrary();
    return PyModule_Create(&cModPyDem);
}
