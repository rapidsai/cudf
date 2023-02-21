# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.table.table cimport table


# TODO: this is vendored from Cython 3.0
cdef extern from "<optional>" namespace "std" nogil:
    cdef cppclass nullopt_t:
        nullopt_t()

    cdef nullopt_t nullopt

    cdef cppclass optional[T]:
        ctypedef T value_type
        optional()
        optional(nullopt_t)
        optional(optional&) except +
        optional(T&) except +
        bool has_value()
        T& value()
        T& value_or[U](U& default_value)
        void swap(optional&)
        void reset()
        T& emplace(...)
        T& operator*()
        optional& operator=(optional&)
        optional& operator=[U](U&)
        bool operator bool()
        bool operator!()
        bool operator==[U](optional&, U&)
        bool operator!=[U](optional&, U&)
        bool operator<[U](optional&, U&)
        bool operator>[U](optional&, U&)
        bool operator<=[U](optional&, U&)
        bool operator>=[U](optional&, U&)

    optional[T] make_optional[T](...) except +


cdef extern from "cudf/io/timezone.hpp" namespace "cudf::io" nogil:
    unique_ptr[table] build_timezone_transition_table(
        optional[string] tzif_dir,
        string timezone_name
    ) except +
