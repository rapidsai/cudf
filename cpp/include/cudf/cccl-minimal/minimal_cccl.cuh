/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Experimental NVRTC-only CCCL subset. Not a public compatibility implementation.
#pragma once
#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif
#ifndef assert
#define assert(X)                    \
  do {                               \
    if (!(X)) asm volatile("trap;"); \
  } while (0)
#endif
using int8_t      = signed char;
using uint8_t     = unsigned char;
using int16_t     = short;
using uint16_t    = unsigned short;
using int32_t     = int;
using uint32_t    = unsigned;
using int64_t     = long;
using uint64_t    = unsigned long;
using __int128_t  = __int128;
using __uint128_t = unsigned __int128;
namespace cuda::std {
using size_t    = unsigned long;
using ptrdiff_t = long;
using nullptr_t = decltype(nullptr);
enum class byte : unsigned char {};
struct alignas(16) max_align_t {
  char bytes[16];
};
using int8_t    = signed char;
using uint8_t   = unsigned char;
using int16_t   = short;
using uint16_t  = unsigned short;
using int32_t   = int;
using uint32_t  = unsigned;
using int64_t   = long;
using uint64_t  = unsigned long;
using intptr_t  = long;
using uintptr_t = unsigned long;
using intmax_t  = long;
using uintmax_t = unsigned long;
#define MINI_INT_ALIAS(N)                \
  using int_fast##N##_t   = int##N##_t;  \
  using uint_fast##N##_t  = uint##N##_t; \
  using int_least##N##_t  = int##N##_t;  \
  using uint_least##N##_t = uint##N##_t;
MINI_INT_ALIAS(8);
MINI_INT_ALIAS(16);
MINI_INT_ALIAS(32);
MINI_INT_ALIAS(64);
#undef MINI_INT_ALIAS
template <class T, T V>
struct integral_constant {
  static constexpr T value = V;
  using value_type         = T;
  constexpr operator T() const { return V; }
  constexpr T operator()() const { return V; }
};
template <bool V>
using bool_constant = integral_constant<bool, V>;
using true_type     = bool_constant<true>;
using false_type    = bool_constant<false>;
template <class A, class B>
struct is_same : false_type {};
template <class A>
struct is_same<A, A> : true_type {};
template <class A, class B>
constexpr bool is_same_v = is_same<A, B>::value;
template <class T>
struct remove_cv {
  using type = T;
};
template <class T>
struct remove_cv<T const> {
  using type = T;
};
template <class T>
struct remove_cv<T volatile> {
  using type = T;
};
template <class T>
struct remove_cv<T const volatile> {
  using type = T;
};
template <class T>
using remove_cv_t = typename remove_cv<T>::type;
template <class T>
struct remove_reference {
  using type = T;
};
template <class T>
struct remove_reference<T&> {
  using type = T;
};
template <class T>
struct remove_reference<T&&> {
  using type = T;
};
template <class T>
using remove_reference_t = typename remove_reference<T>::type;
template <class T>
using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;
template <class T>
T&& declval();
template <class T>
constexpr remove_reference_t<T>&& move(T&& v)
{
  return static_cast<remove_reference_t<T>&&>(v);
}
template <class T>
constexpr T&& forward(remove_reference_t<T>& v)
{
  return static_cast<T&&>(v);
}
template <class T>
constexpr T&& forward(remove_reference_t<T>&& v)
{
  return static_cast<T&&>(v);
}
template <bool B, class T = void>
struct enable_if {};
template <class T>
struct enable_if<true, T> {
  using type = T;
};
template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;
template <bool B, class T, class F>
struct conditional {
  using type = T;
};
template <class T, class F>
struct conditional<false, T, F> {
  using type = F;
};
template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;
template <class...>
using void_t = void;
template <class...>
struct disjunction : false_type {};
template <class H, class... T>
struct disjunction<H, T...> : bool_constant<H::value || disjunction<T...>::value> {};
template <class T>
constexpr bool integral_impl = false;
#define MINI_INT(T) \
  template <>       \
  constexpr bool integral_impl<T> = true;
MINI_INT(bool);
MINI_INT(char);
MINI_INT(signed char);
MINI_INT(unsigned char);
MINI_INT(short);
MINI_INT(unsigned short);
MINI_INT(int);
MINI_INT(unsigned int);
MINI_INT(long);
MINI_INT(unsigned long);
MINI_INT(long long);
MINI_INT(unsigned long long);
MINI_INT(__int128);
MINI_INT(unsigned __int128);
MINI_INT(wchar_t);
MINI_INT(char16_t);
MINI_INT(char32_t);
#undef MINI_INT
template <class T>
struct is_integral : bool_constant<integral_impl<remove_cv_t<T>>> {};
template <class T>
constexpr bool is_integral_v = is_integral<T>::value;
template <class T>
struct is_floating_point
  : bool_constant<is_same_v<remove_cv_t<T>, float> || is_same_v<remove_cv_t<T>, double> ||
                  is_same_v<remove_cv_t<T>, long double>> {};
template <class T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;
template <class T>
struct is_arithmetic : bool_constant<is_integral_v<T> || is_floating_point_v<T>> {};
template <class T>
constexpr bool is_arithmetic_v = is_arithmetic<T>::value;
template <class T, bool = is_arithmetic_v<T>>
struct is_signed : false_type {};
template <class T>
struct is_signed<T, true> : bool_constant<(T(-1) < T(0))> {};
template <class T>
constexpr bool is_signed_v = is_signed<T>::value;
template <class T>
constexpr bool is_unsigned_v = is_integral_v<T> && !is_signed_v<T>;
template <class T>
struct is_unsigned : bool_constant<is_unsigned_v<T>> {};
template <class T>
constexpr bool is_void_v = is_same_v<remove_cv_t<T>, void>;
template <class T>
struct make_unsigned;
template <class T>
struct make_signed;
#define MINI_SIGNED(S, U)   \
  template <>               \
  struct make_unsigned<S> { \
    using type = U;         \
  };                        \
  template <>               \
  struct make_unsigned<U> { \
    using type = U;         \
  };                        \
  template <>               \
  struct make_signed<S> {   \
    using type = S;         \
  };                        \
  template <>               \
  struct make_signed<U> {   \
    using type = S;         \
  };
MINI_SIGNED(signed char, unsigned char);
MINI_SIGNED(short, unsigned short);
MINI_SIGNED(int, unsigned int);
MINI_SIGNED(long, unsigned long);
MINI_SIGNED(long long, unsigned long long);
MINI_SIGNED(__int128, unsigned __int128);
#undef MINI_SIGNED
template <class T>
using make_unsigned_t = typename make_unsigned<remove_cv_t<T>>::type;
template <class T>
using make_signed_t = typename make_signed<remove_cv_t<T>>::type;
template <class... T>
struct common_type;
template <class T>
struct common_type<T> {
  using type = remove_cvref_t<T>;
};
template <class A, class B>
struct common_type<A, B> {
  using type = remove_cvref_t<decltype(true ? declval<A>() : declval<B>())>;
};
template <class A, class B, class... T>
struct common_type<A, B, T...> : common_type<typename common_type<A, B>::type, T...> {};
template <class... T>
using common_type_t = typename common_type<T...>::type;
template <class T>
void conversion_sink(T);
template <class A, class B>
  struct is_convertible : bool_constant < requires {
  conversion_sink<B>(declval<A>());
}>{};
template <>
struct is_convertible<void, void> : true_type {};
template <class A, class B>
concept same_as = is_same_v<A, B>;
template <class T>
concept integral = is_integral_v<T>;
template <class A, class B>
concept equality_comparable_with = requires(A a, B b) {
  a == b;
  a != b;
};
template <class A, class B>
concept totally_ordered_with = equality_comparable_with<A, B> && requires(A a, B b) {
  a < b;
  a > b;
  a <= b;
  a >= b;
};
template <class T, T... V>
struct integer_sequence {
  static constexpr size_t size() { return sizeof...(V); }
};
template <class T, T N, bool Done, T... V>
struct sequence_builder : sequence_builder<T, N - 1, (N - 1 == 0), N - 1, V...> {};
template <class T, T N, T... V>
struct sequence_builder<T, N, true, V...> {
  using type = integer_sequence<T, V...>;
};
template <class T, T N>
using make_integer_sequence = typename sequence_builder<T, N, (N == 0)>::type;
template <size_t... V>
using index_sequence = integer_sequence<size_t, V...>;
template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;
template <class... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;
template <size_t I, class T>
struct tuple_leaf {
  T value;
  constexpr tuple_leaf() = default;
  constexpr tuple_leaf(T v) : value(v) {}
};
template <class S, class... T>
struct tuple_storage;
template <size_t... I, class... T>
struct tuple_storage<index_sequence<I...>, T...> : tuple_leaf<I, T>... {
  constexpr tuple_storage() = default;
  constexpr tuple_storage(T... v)
    requires(sizeof...(T) > 0)
    : tuple_leaf<I, T>(v)...
  {
  }
};
template <class... T>
struct tuple : tuple_storage<index_sequence_for<T...>, T...> {
  using tuple_storage<index_sequence_for<T...>, T...>::tuple_storage;
};
template <class... T>
tuple(T...) -> tuple<T...>;
template <size_t I, class T>
struct tuple_element;
template <class H, class... T>
struct tuple_element<0, tuple<H, T...>> {
  using type = H;
};
template <size_t I, class H, class... T>
struct tuple_element<I, tuple<H, T...>> : tuple_element<I - 1, tuple<T...>> {};
template <size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;
template <size_t I, class... T>
constexpr auto& get(tuple<T...>& t)
{
  return static_cast<tuple_leaf<I, tuple_element_t<I, tuple<T...>>>&>(t).value;
}
template <size_t I, class... T>
constexpr auto const& get(tuple<T...> const& t)
{
  return static_cast<tuple_leaf<I, tuple_element_t<I, tuple<T...>>> const&>(t).value;
}
template <size_t I, class... T>
constexpr auto&& get(tuple<T...>&& t)
{
  return move(get<I>(t));
}
template <class F, class Tuple, size_t... I>
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, index_sequence<I...>)
{
  return forward<F>(f)(get<I>(forward<Tuple>(t))...);
}
template <class F, class... T>
constexpr decltype(auto) apply(F&& f, tuple<T...>& t)
{
  return apply_impl(forward<F>(f), t, index_sequence_for<T...>{});
}
template <class F, class... T>
constexpr decltype(auto) apply(F&& f, tuple<T...> const& t)
{
  return apply_impl(forward<F>(f), t, index_sequence_for<T...>{});
}
template <class F, class... T>
constexpr decltype(auto) apply(F&& f, tuple<T...>&& t)
{
  return apply_impl(forward<F>(f), move(t), index_sequence_for<T...>{});
}
template <class... A, class... B, size_t... I, size_t... J>
constexpr auto cat_impl(tuple<A...> const& a,
                        tuple<B...> const& b,
                        index_sequence<I...>,
                        index_sequence<J...>)
{
  return tuple<A..., B...>{get<I>(a)..., get<J>(b)...};
}
template <class... A, class... B>
constexpr auto tuple_cat(tuple<A...> const& a, tuple<B...> const& b)
{
  return cat_impl(a, b, index_sequence_for<A...>{}, index_sequence_for<B...>{});
}
template <class A, class B>
struct pair {
  A first;
  B second;
};
template <class A, class B>
pair(A, B) -> pair<A, B>;
template <size_t I, class A, class B>
constexpr decltype(auto) get(pair<A, B> const& p)
{
  if constexpr (I == 0)
    return (p.first);
  else
    return (p.second);
}
struct nullopt_t {};
inline constexpr nullopt_t nullopt{};
template <class T>
struct optional {
  using value_type = T;
  T payload{};
  bool valid           = false;
  constexpr optional() = default;
  constexpr optional(nullopt_t) {}
  constexpr optional(T v) : payload(v), valid(true) {}
  constexpr bool has_value() const { return valid; }
  constexpr explicit operator bool() const { return valid; }
  constexpr T& value() { return payload; }
  constexpr T const& value() const { return payload; }
  constexpr T& operator*() { return payload; }
  constexpr T const& operator*() const { return payload; }
  constexpr T* operator->() { return &payload; }
  constexpr T const* operator->() const { return &payload; }
  template <class U>
  constexpr T value_or(U v) const
  {
    return valid ? payload : static_cast<T>(v);
  }
};
template <class T>
optional(T) -> optional<T>;
template <class A, class B>
constexpr bool operator==(optional<A> const& a, optional<B> const& b)
  requires requires { *a == *b; }
{
  return a.has_value() == b.has_value() && (!a.has_value() || *a == *b);
}
template <class A, class B>
constexpr bool operator!=(optional<A> const& a, optional<B> const& b)
  requires requires { *a == *b; }
{
  return !(a == b);
}

template <class E>
struct unexpected {
  E payload;
  constexpr explicit unexpected(E e) : payload(e) {};
};
template <class E>
unexpected(E) -> unexpected<E>;
template <class T, class E>
struct expected {
  using value_type = T;
  T payload{};
  E failure{};
  bool valid           = true;
  constexpr expected() = default;
  constexpr expected(T v) : payload(v) {}
  constexpr expected(unexpected<E> e) : failure(e.payload), valid(false) {}
  constexpr bool has_value() const { return valid; }
  constexpr explicit operator bool() const { return valid; }
  constexpr T& value() { return payload; }
  constexpr T const& value() const { return payload; }
  constexpr T& operator*() { return payload; }
  constexpr T const& operator*() const { return payload; }
  constexpr E error() const { return failure; }
};
template <class T>
struct span {
  T* pointer{};
  size_t length{};
  constexpr span() = default;
  constexpr span(T* p, size_t n) : pointer(p), length(n) {}
  constexpr T* data() const { return pointer; }
  constexpr size_t size() const { return length; }
  constexpr T* begin() const { return pointer; }
  constexpr T* end() const { return pointer + length; }
  constexpr T& operator[](size_t i) const { return pointer[i]; }
};
template <class T>
constexpr T const& min(T const& a, T const& b)
{
  return b < a ? b : a;
}
template <class T>
constexpr T const& max(T const& a, T const& b)
{
  return a < b ? b : a;
}
template <class T>
struct numeric_limits {
  static constexpr bool is_signed = is_signed_v<T>;
  static constexpr int digits     = sizeof(T) * 8 - is_signed;
  static constexpr T max()
  {
    using U = make_unsigned_t<T>;
    U all   = ~U{0};
    return static_cast<T>(all >> is_signed);
  }
  static constexpr T min()
  {
    if constexpr (is_signed)
      return static_cast<T>(make_unsigned_t<T>{1} << digits);
    else
      return T{0};
  }
};
#define MINI_FLOAT_LIMIT(T, D, LO, HI, INF)                            \
  template <>                                                          \
  struct numeric_limits<T> {                                           \
    static constexpr int digits       = D;                             \
    static constexpr bool is_iec559   = true;                          \
    static constexpr int max_exponent = (sizeof(T) == 4 ? 128 : 1024); \
    static constexpr T min() { return LO; }                            \
    static constexpr T max() { return HI; }                            \
    static constexpr T lowest() { return -HI; }                        \
    static __device__ T infinity() { return INF; }                     \
  };
MINI_FLOAT_LIMIT(float, 24, 0x1p-126f, 0x1.fffffep127f, __int_as_float(0x7f800000));
MINI_FLOAT_LIMIT(
  double, 53, 0x1p-1022, 0x1.fffffffffffffp1023, __longlong_as_double(0x7ff0000000000000LL));
#undef MINI_FLOAT_LIMIT
template <class To, class From>
__device__ To bit_cast(From v)
{
  static_assert(sizeof(To) == sizeof(From));
  To out;
  auto* dst       = reinterpret_cast<unsigned char*>(&out);
  auto const* src = reinterpret_cast<unsigned char const*>(&v);
  for (size_t i = 0; i < sizeof(To); ++i)
    dst[i] = src[i];
  return out;
}
template <class T>
__device__ int countl_zero(T v)
{
  if constexpr (sizeof(T) <= 4)
    return v ? __clz(static_cast<unsigned>(v)) - (32 - sizeof(T) * 8) : sizeof(T) * 8;
  else
    return v ? __clzll(static_cast<unsigned long long>(v)) : 64;
}
template <class T>
__device__ int countr_zero(T v)
{
  if constexpr (sizeof(T) <= 4)
    return v ? __ffs(static_cast<int>(v)) - 1 : sizeof(T) * 8;
  else
    return v ? __ffsll(static_cast<long long>(v)) - 1 : 64;
}
template <class T>
__device__ T abs(T v)
  requires(is_integral_v<T>)
{
  return v < 0 ? static_cast<T>(-static_cast<make_unsigned_t<T>>(v)) : v;
}
#define MINI_MATH(F)                                       \
  __device__ inline float F(float x) { return ::F##f(x); } \
  __device__ inline double F(double x) { return ::F(x); }
MINI_MATH(fabs);
MINI_MATH(floor);
MINI_MATH(ceil);
MINI_MATH(rint);
MINI_MATH(sqrt);
MINI_MATH(cbrt);
MINI_MATH(exp);
MINI_MATH(log);
MINI_MATH(acos);
MINI_MATH(acosh);
MINI_MATH(asin);
MINI_MATH(asinh);
MINI_MATH(atan);
MINI_MATH(atanh);
MINI_MATH(cos);
MINI_MATH(cosh);
MINI_MATH(sin);
MINI_MATH(sinh);
MINI_MATH(tan);
MINI_MATH(tanh);
#undef MINI_MATH
__device__ inline float abs(float x) { return ::fabsf(x); }
__device__ inline double abs(double x) { return ::fabs(x); }
__device__ inline float fmod(float a, float b) { return ::fmodf(a, b); }
__device__ inline double fmod(double a, double b) { return ::fmod(a, b); }
__device__ inline float pow(float a, float b) { return ::powf(a, b); }
__device__ inline double pow(double a, double b) { return ::pow(a, b); }
template <class A, class B>
__device__ double fmod(A a, B b)
  requires(is_arithmetic_v<A> && is_arithmetic_v<B> &&
           (!is_same_v<A, B> || is_integral_v<A>))
{
  return ::fmod(static_cast<double>(a), static_cast<double>(b));
}
template <class A, class B>
__device__ double pow(A a, B b)
  requires(is_arithmetic_v<A> && is_arithmetic_v<B> &&
           (!is_same_v<A, B> || is_integral_v<A>))
{
  return ::pow(static_cast<double>(a), static_cast<double>(b));
}
struct input_iterator_tag {};
struct random_access_iterator_tag : input_iterator_tag {};
template <class I>
struct iterator_traits {
  using value_type      = typename I::value_type;
  using difference_type = typename I::difference_type;
};
template <class T>
struct iterator_traits<T*> {
  using value_type      = remove_cv_t<T>;
  using difference_type = ptrdiff_t;
};
template <class I>
constexpr auto distance(I a, I b)
{
  return b - a;
}
template <long long N, long long D = 1>
struct ratio {
  static constexpr long long num = N, den = D;
};
namespace chrono {
constexpr long long gcd(long long a, long long b) { return b ? gcd(b, a % b) : a; }
template <class R, class P = ratio<1>>
struct duration {
  using rep    = R;
  using period = P;
  R ticks{};
  constexpr duration() = default;
  constexpr explicit duration(R v) : ticks(v) {}
  template <class U, class Q>
  constexpr duration(duration<U, Q> v)
  {
    constexpr auto numerator = Q::num * P::den, denominator = Q::den * P::num;
    constexpr auto factor = gcd(numerator, denominator);
    using C               = common_type_t<R, U, long long>;
    ticks = static_cast<R>(C(v.count()) * (numerator / factor) / (denominator / factor));
  }
  constexpr R count() const { return ticks; }
  constexpr duration operator-() const { return duration{-ticks}; }
};
using days         = duration<int, ratio<86400>>;
using hours        = duration<int, ratio<3600>>;
using minutes      = duration<int, ratio<60>>;
using seconds      = duration<long long>;
using milliseconds = duration<long long, ratio<1, 1000>>;
using microseconds = duration<long long, ratio<1, 1000000>>;
using nanoseconds  = duration<long long, ratio<1, 1000000000>>;
struct system_clock {};
template <class Clock, class D>
struct time_point {
  using duration = D;
  using rep      = typename D::rep;
  using period   = typename D::period;
  D ticks{};
  constexpr time_point() = default;
  constexpr explicit time_point(D d) : ticks(d) {}
  template <class E>
  constexpr time_point(time_point<Clock, E> t) : ticks(t.time_since_epoch())
  {
  }
  constexpr D time_since_epoch() const { return ticks; }
};
template <class D>
using sys_time = time_point<system_clock, D>;

template <class A, class P, class B, class Q>
using duration_common =
  duration<common_type_t<A, B>,
           ratio<gcd(P::num, Q::num), (P::den / gcd(P::den, Q::den)) * Q::den>>;
#define MINI_DURATION_OP(OP)                                     \
  template <class A, class P, class B, class Q>                  \
  constexpr auto operator OP(duration<A, P> a, duration<B, Q> b) \
  {                                                              \
    using D = duration_common<A, P, B, Q>;                       \
    return D{D(a).count() OP D(b).count()};                      \
  }
MINI_DURATION_OP(+);
MINI_DURATION_OP(-);
MINI_DURATION_OP(%);
#undef MINI_DURATION_OP
#define MINI_DURATION_CMP(OP)                                        \
  template <class A, class P, class B, class Q>                      \
  constexpr bool operator OP(duration<A, P> a, duration<B, Q> b)     \
  {                                                                  \
    using D = duration_common<A, P, B, Q>;                           \
    return D(a).count() OP D(b).count();                             \
  }                                                                  \
  template <class C, class A, class B>                               \
  constexpr bool operator OP(time_point<C, A> a, time_point<C, B> b) \
  {                                                                  \
    return a.time_since_epoch() OP b.time_since_epoch();             \
  }
MINI_DURATION_CMP(==);
MINI_DURATION_CMP(!=);
MINI_DURATION_CMP(<);
MINI_DURATION_CMP(>);
MINI_DURATION_CMP(<=);
MINI_DURATION_CMP(>=);
#undef MINI_DURATION_CMP
template <class A, class P, class B>
constexpr auto operator*(duration<A, P> a, B b)
  requires(is_arithmetic_v<B>)
{
  return duration<common_type_t<A, B>, P>{a.count() * b};
}
template <class A, class P, class B>
constexpr auto operator*(B b, duration<A, P> a)
  requires(is_arithmetic_v<B>)
{
  return a * b;
}
template <class A, class P, class B>
constexpr auto operator/(duration<A, P> a, B b)
  requires(is_arithmetic_v<B>)
{
  return duration<common_type_t<A, B>, P>{a.count() / b};
}
template <class A, class P, class B, class Q>
constexpr auto operator/(duration<A, P> a, duration<B, Q> b)
{
  using D = duration_common<A, P, B, Q>;
  return D(a).count() / D(b).count();
}
template <class C, class A, class B>
constexpr auto operator-(time_point<C, A> a, time_point<C, B> b)
{
  return a.time_since_epoch() - b.time_since_epoch();
}
template <class C, class A, class B, class P>
constexpr auto operator+(time_point<C, A> a, duration<B, P> b)
{
  auto d = a.time_since_epoch() + b;
  return time_point<C, decltype(d)>{d};
}
template <class C, class A, class B, class P>
constexpr auto operator-(time_point<C, A> a, duration<B, P> b)
{
  auto d = a.time_since_epoch() - b;
  return time_point<C, decltype(d)>{d};
}
template <class D, class A, class P>
constexpr D duration_cast(duration<A, P> a)
{
  return D(a);
}
}  // namespace chrono
}  // namespace cuda::std
namespace cuda {
enum thread_scope { thread_scope_device };
enum memory_order { memory_order_relaxed };
template <class T, thread_scope Scope = thread_scope_device>
struct atomic_ref {
  T* address;
  __device__ explicit atomic_ref(T& v) : address(&v) {}
  __device__ T fetch_or(T v, memory_order) { return atomicOr(address, v); }
  __device__ T fetch_and(T v, memory_order) { return atomicAnd(address, v); }
  __device__ T fetch_max(T v, memory_order) { return atomicMax(address, v); }
};
template <class T>
atomic_ref(T&) -> atomic_ref<T>;
template <class T>
constexpr T bitmask(int start, int count)
{
  constexpr int bits = sizeof(T) * 8;
  return (count == bits ? ~T{0} : ((T{1} << count) - 1)) << start;
}
template <class T>
constexpr T ceil_div(T a, T b)
{
  return a / b + (a % b != 0);
}
namespace ptx {
__device__ inline bool elect_sync(unsigned mask)
{
#if __CUDA_ARCH__ >= 900
  unsigned result;
  asm volatile("{ .reg .pred p; elect.sync _|p, %1; selp.u32 %0, 1, 0, p; }"
               : "=r"(result)
               : "r"(mask));
  return result;
#else
  return (threadIdx.x & 31) == (__ffs(mask) - 1);
#endif
}
}  // namespace ptx
template <class T>
struct overflow_result {
  T value;
  bool overflow;
};
template <class T>
__device__ bool add_overflow(T& out, T a, T b)
{
  using U = std::make_unsigned_t<T>;
  out     = static_cast<T>(U(a) + U(b));
  if constexpr (std::is_signed_v<T>)
    return (b > 0 && a > std::numeric_limits<T>::max() - b) ||
           (b < 0 && a < std::numeric_limits<T>::min() - b);
  else
    return out < a;
}
template <class T>
__device__ bool sub_overflow(T& out, T a, T b)
{
  using U = std::make_unsigned_t<T>;
  out     = static_cast<T>(U(a) - U(b));
  if constexpr (std::is_signed_v<T>)
    return (b < 0 && a > std::numeric_limits<T>::max() + b) ||
           (b > 0 && a < std::numeric_limits<T>::min() + b);
  else
    return a < b;
}
template <class T>
__device__ bool mul_overflow(T& out, T a, T b)
{
  using U = std::make_unsigned_t<T>;
  using W = std::conditional_t<(sizeof(U) < sizeof(unsigned)), unsigned, U>;
  out     = static_cast<T>(W(U(a)) * W(U(b)));
  if (a == 0 || b == 0) return false;
  auto lo = std::numeric_limits<T>::min();
  auto hi = std::numeric_limits<T>::max();
  if constexpr (std::is_signed_v<T>)
    return a > 0 ? (b > 0 ? a > hi / b : b < lo / a) : (b > 0 ? a < lo / b : a < hi / b);
  else
    return a > hi / b;
}
template <class T>
__device__ bool div_overflow(T& out, T a, T b)
{
  if (b == 0) {
    out = 0;
    return true;
  }
  if constexpr (std::is_signed_v<T>) {
    if (a == std::numeric_limits<T>::min() && b == T(-1)) {
      out = a;
      return true;
    }
  }
  out = a / b;
  return false;
}
#define MINI_OVERFLOW(F)                    \
  template <class T>                        \
  __device__ overflow_result<T> F(T a, T b) \
  {                                         \
    T out{};                                \
    bool bad = F(out, a, b);                \
    return {out, bad};                      \
  }
MINI_OVERFLOW(add_overflow);
MINI_OVERFLOW(sub_overflow);
MINI_OVERFLOW(mul_overflow);
MINI_OVERFLOW(div_overflow);
#undef MINI_OVERFLOW
}  // namespace cuda
namespace cuda::std {
using ::cuda::memory_order;
using ::cuda::memory_order_relaxed;
}  // namespace cuda::std
