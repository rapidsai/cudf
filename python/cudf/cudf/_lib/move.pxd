# Copyright (c) 2020, NVIDIA CORPORATION.

# This whole file should be deleted when upgrading to Cython >= 3
# See https://github.com/cython/cython/pull/3358
cdef extern from * namespace "cython_std" nogil:
    """
    #if __cplusplus > 199711L
    #include <type_traits>
    namespace cython_std {
    template <typename T> typename std::remove_reference<T>::type&&
    move(T& t) noexcept { return std::move(t); }
    template <typename T> typename std::remove_reference<T>::type&&
    move(T&& t) noexcept { return std::move(t); }
    }
    #endif
    """
    cdef T move[T](T)
