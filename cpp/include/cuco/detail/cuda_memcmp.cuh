__host__ __device__
inline int cuda_memcmp(void const * __lhs, void const * __rhs, size_t __count) {
    auto __lhs_c = reinterpret_cast<unsigned char const *>(__lhs);
    auto __rhs_c = reinterpret_cast<unsigned char const *>(__rhs);
    while (__count--) {
        auto const __lhs_v = *__lhs_c++;
        auto const __rhs_v = *__rhs_c++;
        if (__lhs_v < __rhs_v) { return -1; }
        if (__lhs_v > __rhs_v) { return 1; }
    }
    return 0;
}