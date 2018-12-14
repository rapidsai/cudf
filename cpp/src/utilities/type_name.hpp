#ifndef CUDF_UTIL_TYPE_NAME_HPP_
#define CUDF_UTIL_TYPE_NAME_HPP_

#include <cstddef>
#include <stdexcept>
#include <cstring>
#include <ostream>

namespace cudf {
namespace util {

class static_string
{
    const char* const p_;
    const std::size_t sz_;

public:
    typedef const char* const_iterator;

    template <std::size_t N>
    constexpr static_string(const char(&a)[N]) noexcept
        : p_(a)
        , sz_(N-1)
        {}

    constexpr static_string(const char* p, std::size_t N) noexcept
        : p_(p)
        , sz_(N)
        {}

    constexpr const char* data() const noexcept {return p_;}
    constexpr std::size_t size() const noexcept {return sz_;}

    constexpr const_iterator begin() const noexcept {return p_;}
    constexpr const_iterator end()   const noexcept {return p_ + sz_;}

    constexpr char operator[](std::size_t n) const
    {
        return n < sz_ ? p_[n] : throw std::out_of_range("static_string");
    }
};

// Not used for getting type names, it's just for completing
// the usability of static_string
inline
std::ostream&
operator<<(std::ostream& os, static_string const& s)
{
    return os.write(s.data(), s.size());
}

/**
 * @brief return the string name of a type.
 *
 * @note This is an
 * alternative to using type_info<T>.name() which also
 * preserves CV qualifiers (const, volatile, reference,
 *  rvalue-reference)
 *
 * The code was copied from this StackOverflow answer:
 *  http://stackoverflow.com/a/20170989/1593077
 * and is due to Howard Hinnant
 */
template <class T>
constexpr
static_string
type_name()
{
#ifdef __clang__
    static_string p = __PRETTY_FUNCTION__;
    return static_string(p.data() + 31, p.size() - 31 - 1);
#elif defined(__GNUC__)
    static_string p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
    return static_string(p.data() + 36, p.size() - 36 - 1);
#  else
    return static_string(p.data() + 46, p.size() - 46 - 1);
#  endif
#elif defined(_MSC_VER)
    static_string p = __FUNCSIG__;
    return static_string(p.data() + 38, p.size() - 38 - 7);
#endif
}

/**
 * This is a convenience macro, so that instead of
 *
 *  type_name<decltype(my_value)>()
 *
 * you could use:
 *
 *   type_name_of(my_value)
 *
 * @param v a value which is only passed to indicate a type
 * @return the string type name of typeof(v)
 */
#define type_name_of(_v) type_name<decltype(_v)>();


} // namespace util
} // namespace cudf

#endif // CUDF_UTIL_TYPE_NAME_HPP_
