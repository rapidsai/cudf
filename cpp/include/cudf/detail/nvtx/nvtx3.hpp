/*
 *  Copyright (c) 2020, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#pragma once

#if defined(NVTX3_MINOR_VERSION) and NVTX3_MINOR_VERSION < 0
#error \
  "Trying to #include NVTX version 3 in a source file where an older NVTX version has already been included.  If you are not directly using NVTX (the NVIDIA Tools Extension library), you are getting this error because libraries you are using have included different versions of NVTX.  Suggested solutions are: (1) reorder #includes so the newest NVTX version is included first, (2) avoid using the conflicting libraries in the same .c/.cpp file, or (3) update the library using the older NVTX version to use the newer version instead."
#endif

/**
 * @brief Semantic minor version number.
 *
 * Major version number is hardcoded into the "nvtx3" namespace/prefix.
 *
 * If this value is incremented, the above version include guard needs to be
 * updated.
 *
 */
#define NVTX3_MINOR_VERSION 0

#include <nvtx3/nvToolsExt.h>

#include <string>

/**
 * @file nvtx3.hpp
 *
 * @brief Provides C++ constructs making the NVTX library safer and easier to
 * use with zero overhead.
 */

/**
 * \mainpage
 * \tableofcontents
 *
 * \section QUICK_START Quick Start
 *
 * To add NVTX ranges to your code, use the `nvtx3::thread_range` RAII object. A
 * range begins when the object is created, and ends when the object is
 * destroyed.
 *
 * \code{.cpp}
 * #include "nvtx3.hpp"
 * void some_function(){
 *    // Begins a NVTX range with the messsage "some_function"
 *    // The range ends when some_function() returns and `r` is destroyed
 *    nvtx3::thread_range r{"some_function"};
 *
 *    for(int i = 0; i < 6; ++i){
 *       nvtx3::thread_range loop{"loop range"};
 *       std::this_thread::sleep_for(std::chrono::seconds{1});
 *    }
 * } // Range ends when `r` is destroyed
 * \endcode
 *
 * The example code above generates the following timeline view in Nsight
 * Systems:
 *
 * \image html
 * https://raw.githubusercontent.com/jrhemstad/nvtx_wrappers/master/docs/example_range.png
 *
 * Alternatively, use the \ref MACROS like `NVTX3_FUNC_RANGE()` to add
 * ranges to your code that automatically use the name of the enclosing function
 * as the range's message.
 *
 * \code{.cpp}
 * #include "nvtx3.hpp"
 * void some_function(){
 *    // Creates a range with a message "some_function" that ends when the
 * enclosing
 *    // function returns
 *    NVTX3_FUNC_RANGE();
 *    ...
 * }
 * \endcode
 *
 *
 * \section Overview
 *
 * The NVTX library provides a set of functions for users to annotate their code
 * to aid in performance profiling and optimization. These annotations provide
 * information to tools like Nsight Systems to improve visualization of
 * application timelines.
 *
 * \ref RANGES are one of the most commonly used NVTX constructs for annotating
 * a span of time. For example, imagine a user wanted to see every time a
 * function, `my_function`, is called and how long it takes to execute. This can
 * be accomplished with an NVTX range created on the entry to the function and
 * terminated on return from `my_function` using the push/pop C APIs:
 *
 * ```
 * void my_function(...){
 *    nvtxRangePushA("my_function"); // Begins NVTX range
 *    // do work
 *    nvtxRangePop(); // Ends NVTX range
 * }
 * ```
 *
 * One of the challenges with using the NVTX C API is that it requires manually
 * terminating the end of the range with `nvtxRangePop`. This can be challenging
 * if `my_function()` has multiple returns or can throw exceptions as it
 * requires calling `nvtxRangePop()` before all possible return points.
 *
 * NVTX++ solves this inconvenience through the "RAII" technique by providing a
 * `nvtx3::thread_range` class that begins a range at construction and ends the
 * range on destruction. The above example then becomes:
 *
 * ```
 * void my_function(...){
 *    nvtx3::thread_range r{"my_function"}; // Begins NVTX range
 *    // do work
 * } // Range ends on exit from `my_function` when `r` is destroyed
 * ```
 *
 * The range object `r` is deterministically destroyed whenever `my_function`
 * returns---ending the NVTX range without manual intervention. For more
 * information, see \ref RANGES and `nvtx3::domain_thread_range`.
 *
 * Another inconvenience of the NVTX C APIs are the several constructs where the
 * user is expected to initialize an object at the beginning of an application
 * and reuse that object throughout the lifetime of the application. For example
 * Domains, Categories, and Registered messages.
 *
 * Example:
 * ```
 * nvtxDomainHandle_t D = nvtxDomainCreateA("my domain");
 * // Reuse `D` throughout the rest of the application
 * ```
 *
 * This can be problematic if the user application or library does not have an
 * explicit initialization function called before all other functions to
 * ensure that these long-lived objects are initialized before being used.
 *
 * NVTX++ makes use of the "construct on first use" technique to alleviate this
 * inconvenience. In short, a function local static object is constructed upon
 * the first invocation of a function and returns a reference to that object on
 * all future invocations. See the documentation for
 * `nvtx3::registered_message`, `nvtx3::domain`, `nvtx3::named_category`,  and
 * https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use for more
 * information.
 *
 * Using construct on first use, the above example becomes:
 * ```
 * struct my_domain{ static constexpr char const* name{"my domain"}; };
 *
 * // The first invocation of `domain::get` for the type `my_domain` will
 * // construct a `nvtx3::domain` object and return a reference to it. Future
 * // invocations simply return a reference.
 * nvtx3::domain const& D = nvtx3::domain::get<my_domain>();
 * ```
 * For more information about NVTX and how it can be used, see
 * https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx and
 * https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
 * for more information.
 *
 * \section RANGES Ranges
 *
 * Ranges are used to describe a span of time during the execution of an
 * application. Common examples are using ranges to annotate the time it takes
 * to execute a function or an iteration of a loop.
 *
 * NVTX++ uses RAII to automate the generation of ranges that are tied to the
 * lifetime of objects. Similar to `std::lock_guard` in the C++ Standard
 * Template Library.
 *
 * \subsection THREAD_RANGE Thread Range
 *
 * `nvtx3::domain_thread_range` is a class that begins a range upon construction
 * and ends the range at destruction. This is one of the most commonly used
 * constructs in NVTX++ and is useful for annotating spans of time on a
 * particular thread. These ranges can be nested to arbitrary depths.
 *
 * `nvtx3::thread_range` is an alias for a `nvtx3::domain_thread_range` in the
 * global NVTX domain. For more information about Domains, see \ref DOMAINS.
 *
 * Various attributes of a range can be configured constructing a
 * `nvtx3::domain_thread_range` with a `nvtx3::event_attributes` object. For
 * more information, see \ref ATTRIBUTES.
 *
 * Example:
 *
 * \code{.cpp}
 * void some_function(){
 *    // Creates a range for the duration of `some_function`
 *    nvtx3::thread_range r{};
 *
 *    while(true){
 *       // Creates a range for every loop iteration
 *       // `loop_range` is nested inside `r`
 *       nvtx3::thread_range loop_range{};
 *    }
 * }
 * \endcode
 *
 * \subsection PROCESS_RANGE Process Range
 *
 * `nvtx3::domain_process_range` is identical to `nvtx3::domain_thread_range`
 * with the exception that a `domain_process_range` can be created and destroyed
 * on different threads. This is useful to annotate spans of time that can
 * bridge multiple threads.
 *
 * `nvtx3::domain_thread_range`s should be preferred unless one needs the
 * ability to begin and end a range on different threads.
 *
 * \section MARKS Marks
 *
 * `nvtx3::mark` allows annotating an instantaneous event in an application's
 * timeline. For example, indicating when a mutex is locked or unlocked.
 *
 * \code{.cpp}
 * std::mutex global_lock;
 * void lock_mutex(){
 *    global_lock.lock();
 *    // Marks an event immediately after the mutex is locked
 *    nvtx3::mark<my_domain>("lock_mutex");
 * }
 * \endcode
 *
 * \section DOMAINS Domains
 *
 * Similar to C++ namespaces, Domains allow for scoping NVTX events. By default,
 * all NVTX events belong to the "global" domain. Libraries and applications
 * should scope their events to use a custom domain to differentiate where the
 * events originate from.
 *
 * It is common for a library or application to have only a single domain and
 * for the name of that domain to be known at compile time. Therefore, Domains
 * in NVTX++ are represented by _tag types_.
 *
 * For example, to define a custom  domain, simply define a new concrete type
 * (a `class` or `struct`) with a `static` member called `name` that contains
 * the desired name of the domain.
 *
 * ```
 * struct my_domain{ static constexpr char const* name{"my domain"}; };
 * ```
 *
 * For any NVTX++ construct that can be scoped to a domain, the type `my_domain`
 * can be passed as an explicit template argument to scope it to the custom
 * domain.
 *
 * The tag type `nvtx3::domain::global` represents the global NVTX domain.
 *
 * \code{.cpp}
 * // By default, `domain_thread_range` belongs to the global domain
 * nvtx3::domain_thread_range<> r0{};
 *
 * // Alias for a `domain_thread_range` in the global domain
 * nvtx3::thread_range r1{};
 *
 * // `r` belongs to the custom domain
 * nvtx3::domain_thread_range<my_domain> r{};
 * \endcode
 *
 * When using a custom domain, it is reccomended to define type aliases for NVTX
 * constructs in the custom domain.
 * ```
 * using my_thread_range = nvtx3::domain_thread_range<my_domain>;
 * using my_registered_message = nvtx3::registered_message<my_domain>;
 * using my_named_category = nvtx3::named_category<my_domain>;
 * ```
 *
 * See `nvtx3::domain` for more information.
 *
 * \section ATTRIBUTES Event Attributes
 *
 * NVTX events can be customized with various attributes to provide additional
 * information (such as a custom message) or to control visualization of the
 * event (such as the color used). These attributes can be specified per-event
 * via arguments to a `nvtx3::event_attributes` object.
 *
 * NVTX events can be customized via four "attributes":
 * - \ref COLOR : color used to visualize the event in tools.
 * - \ref MESSAGES :  Custom message string.
 * - \ref PAYLOAD :  User-defined numerical value.
 * - \ref CATEGORY : Intra-domain grouping.
 *
 * It is possible to construct a `nvtx3::event_attributes` from any number of
 * attribute objects (nvtx3::color, nvtx3::message, nvtx3::payload,
 * nvtx3::category) in any order. If an attribute is not specified, a tool
 * specific default value is used. See `nvtx3::event_attributes` for more
 * information.
 *
 * \code{.cpp}
 * // Custom color, message
 * event_attributes attr{nvtx3::rgb{127, 255, 0},
 *                      "message"};
 *
 * // Custom color, message, payload, category
 * event_attributes attr{nvtx3::rgb{127, 255, 0},
 *                      nvtx3::payload{42},
 *                      "message",
 *                      nvtx3::category{1}};
 *
 * // Arguments can be in any order
 * event_attributes attr{nvtx3::payload{42},
 *                      nvtx3::category{1},
 *                      "message",
 *                      nvtx3::rgb{127, 255, 0}};
 *
 * // "First wins" with multiple arguments of the same type
 * event_attributes attr{ nvtx3::payload{42}, nvtx3::payload{7} }; // payload is
 * 42 \endcode
 *
 * \subsection MESSAGES message
 *
 * A `nvtx3::message` allows associating a custom message string with an NVTX
 * event.
 *
 * Example:
 * \code{.cpp}
 * // Create an `event_attributes` with the custom message "my message"
 * nvtx3::event_attributes attr{nvtx3::Mesage{"my message"}};
 *
 * // strings and string literals implicitly assumed to be a `nvtx3::message`
 * nvtx3::event_attributes attr{"my message"};
 * \endcode
 *
 * \subsubsection REGISTERED_MESSAGE Registered Messages
 *
 * Associating a `nvtx3::message` with an event requires copying the contents of
 * the message every time the message is used, i.e., copying the entire message
 * string. This may cause non-trivial overhead in performance sensitive code.
 *
 * To eliminate this overhead, NVTX allows registering a message string,
 * yielding a "handle" that is inexpensive to copy that may be used in place of
 * a message string. When visualizing the events, tools such as Nsight Systems
 * will take care of mapping the message handle to its string.
 *
 * A message should be registered once and the handle reused throughout the rest
 * of the application. This can be done by either explicitly creating static
 * `nvtx3::registered_message` objects, or using the
 * `nvtx3::registered_message::get` construct on first use helper (recommended).
 *
 * Similar to \ref DOMAINS, `nvtx3::registered_message::get` requires defining a
 * custom tag type with a static `message` member whose value will be the
 * contents of the registered string.
 *
 * Example:
 * \code{.cpp}
 * // Explicitly constructed, static `registered_message`
 * static registered_message<my_domain> static_message{"my message"};
 *
 * // Or use construct on first use:
 * // Define a tag type with a `message` member string to register
 * struct my_message{ static constexpr char const* message{ "my message" }; };
 *
 * // Uses construct on first use to register the contents of
 * // `my_message::message`
 * nvtx3::registered_message<my_domain> const& msg =
 * nvtx3::registered_message<my_domain>::get<my_message>(); \endcode
 *
 * \subsection COLOR color
 *
 * Associating a `nvtx3::color` with an event allows controlling how the event
 * is visualized in a tool such as Nsight Systems. This is a convenient way to
 * visually differentiate among different events.
 *
 * \code{.cpp}
 * // Define a color via rgb color values
 * nvtx3::color c{nvtx3::rgb{127, 255, 0}};
 * nvtx3::event_attributes attr{c};
 *
 * // rgb color values can be passed directly to an `event_attributes`
 * nvtx3::event_attributes attr1{nvtx3::rgb{127,255,0}};
 * \endcode
 *
 * \subsection CATEGORY category
 *
 * A `nvtx3::category` is simply an integer id that allows for fine-grain
 * grouping of NVTX events. For example, one might use separate categories for
 * IO, memory allocation, compute, etc.
 *
 * \code{.cpp}
 * nvtx3::event_attributes{nvtx3::category{1}};
 * \endcode
 *
 * \subsubsection NAMED_CATEGORIES Named Categories
 *
 * Associates a `name` string with a category `id` to help differentiate among
 * categories.
 *
 * For any given category id `Id`, a `named_category{Id, "name"}` should only
 * be constructed once and reused throughout an application. This can be done by
 * either explicitly creating static `nvtx3::named_category` objects, or using
 * the `nvtx3::named_category::get` construct on first use helper (recommended).
 *
 * Similar to \ref DOMAINS, `nvtx3::named_category::get` requires defining a
 * custom tag type with static `name` and `id` members.
 *
 * \code{.cpp}
 * // Explicitly constructed, static `named_category`
 * static nvtx3::named_category static_category{42, "my category"};
 *
 * // OR use construct on first use:
 * // Define a tag type with `name` and `id` members
 * struct my_category{
 *    static constexpr char const* name{"my category"}; // category name
 *    static constexpr category::id_type id{42}; // category id
 * };
 *
 * // Use construct on first use to name the category id `42`
 * // with name "my category"
 * nvtx3::named_category const& my_category =
 * named_category<my_domain>::get<my_category>();
 *
 * // Range `r` associated with category id `42`
 * nvtx3::event_attributes attr{my_category};
 * \endcode
 *
 * \subsection PAYLOAD payload
 *
 * Allows associating a user-defined numerical value with an event.
 *
 * ```
 * nvtx3:: event_attributes attr{nvtx3::payload{42}}; // Constructs a payload
 * from
 *                                                 // the `int32_t` value 42
 * ```
 *
 *
 * \section EXAMPLE Example
 *
 * Putting it all together:
 * \code{.cpp}
 * // Define a custom domain tag type
 * struct my_domain{ static constexpr char const* name{"my domain"}; };
 *
 * // Define a named category tag type
 * struct my_category{
 *    static constexpr char const* name{"my category"};
 *    static constexpr uint32_t id{42};
 * };
 *
 * // Define a registered message tag type
 * struct my_message{ static constexpr char const* message{"my message"}; };
 *
 * // For convenience, use aliases for domain scoped objects
 * using my_thread_range = nvtx3::domain_thread_range<my_domain>;
 * using my_registered_message = nvtx3::registered_message<my_domain>;
 * using my_named_category = nvtx3::named_category<my_domain>;
 *
 * // Default values for all attributes
 * nvtx3::event_attributes attr{};
 * my_thread_range r0{attr};
 *
 * // Custom (unregistered) message, and unnamed category
 * nvtx3::event_attributes attr1{"message", nvtx3::category{2}};
 * my_thread_range r1{attr1};
 *
 * // Alternatively, pass arguments of `event_attributes` ctor directly to
 * // `my_thread_range`
 * my_thread_range r2{"message", nvtx3::category{2}};
 *
 * // construct on first use a registered message
 * auto msg = my_registered_message::get<my_message>();
 *
 * // construct on first use a named category
 * auto category = my_named_category::get<my_category>();
 *
 * // Use registered message and named category
 * my_thread_range r3{msg, category, nvtx3::rgb{127, 255, 0},
 *                    nvtx3::payload{42}};
 *
 * // Any number of arguments in any order
 * my_thread_range r{nvtx3::rgb{127, 255,0}, msg};
 *
 * \endcode
 * \section MACROS Convenience Macros
 *
 * Oftentimes users want to quickly and easily add NVTX ranges to their library
 * or application to aid in profiling and optimization.
 *
 * A convenient way to do this is to use the \ref NVTX3_FUNC_RANGE and
 * \ref NVTX3_FUNC_RANGE_IN macros. These macros take care of constructing an
 * `nvtx3::domain_thread_range` with the name of the enclosing function as the
 * range's message.
 *
 * \code{.cpp}
 * void some_function(){
 *    // Automatically generates an NVTX range for the duration of the function
 *    // using "some_function" as the event's message.
 *    NVTX3_FUNC_RANGE();
 * }
 * \endcode
 *
 */

/**
 * @brief Enables the use of constexpr when support for C++14 relaxed constexpr
 * is present.
 *
 * Initializing a legacy-C (i.e., no constructor) union member requires
 * initializing in the constructor body. Non-empty constexpr constructors
 * require C++14 relaxed constexpr.
 *
 */
#if __cpp_constexpr >= 201304L
#define NVTX3_RELAXED_CONSTEXPR constexpr
#else
#define NVTX3_RELAXED_CONSTEXPR
#endif

namespace nvtx3 {
namespace detail {
/**
 * @brief Verifies if a type `T` contains a member `T::name` of type `const
 * char*` or `const wchar_t*`.
 *
 * @tparam T The type to verify
 * @return True if `T` contains a member `T::name` of type `const char*` or
 * `const wchar_t*`.
 */
template <typename T>
constexpr auto has_name_member() noexcept -> decltype(T::name, bool())
{
  return (std::is_same<char const*, typename std::decay<decltype(T::name)>::type>::value or
          std::is_same<wchar_t const*, typename std::decay<decltype(T::name)>::type>::value);
}
}  // namespace detail

/**
 * @brief `domain`s allow for grouping NVTX events into a single scope to
 * differentiate them from events in other `domain`s.
 *
 * By default, all NVTX constructs are placed in the "global" NVTX domain.
 *
 * A custom `domain` may be used in order to differentiate a library's or
 * application's NVTX events from other events.
 *
 * `domain`s are expected to be long-lived and unique to a library or
 * application. As such, it is assumed a domain's name is known at compile
 * time. Therefore, all NVTX constructs that can be associated with a domain
 * require the domain to be specified via a *type* `DomainName` passed as an
 * explicit template parameter.
 *
 * The type `domain::global` may be used to indicate that the global NVTX
 * domain should be used.
 *
 * None of the C++ NVTX constructs require the user to manually construct a
 * `domain` object. Instead, if a custom domain is desired, the user is
 * expected to define a type `DomainName` that contains a member
 * `DomainName::name` which resolves to either a `char const*` or `wchar_t
 * const*`. The value of `DomainName::name` is used to name and uniquely
 * identify the custom domain.
 *
 * Upon the first use of an NVTX construct associated with the type
 * `DomainName`, the "construct on first use" pattern is used to construct a
 * function local static `domain` object. All future NVTX constructs
 * associated with `DomainType` will use a reference to the previously
 * constructed `domain` object. See `domain::get`.
 *
 * Example:
 * ```
 * // The type `my_domain` defines a `name` member used to name and identify
 * the
 * // `domain` object identified by `my_domain`.
 * struct my_domain{ static constexpr char const* name{"my_domain"}; };
 *
 * // The NVTX range `r` will be grouped with all other NVTX constructs
 * // associated with  `my_domain`.
 * nvtx3::domain_thread_range<my_domain> r{};
 *
 * // An alias can be created for a `domain_thread_range` in the custom domain
 * using my_thread_range = nvtx3::domain_thread_range<my_domain>;
 * my_thread_range my_range{};
 *
 * // `domain::global` indicates that the global NVTX domain is used
 * nvtx3::domain_thread_range<domain::global> r2{};
 *
 * // For convenience, `nvtx3::thread_range` is an alias for a range in the
 * // global domain
 * nvtx3::thread_range r3{};
 * ```
 */
class domain {
 public:
  domain(domain const&) = delete;
  domain& operator=(domain const&) = delete;
  domain(domain&&)                 = delete;
  domain& operator=(domain&&) = delete;

  /**
   * @brief Returns reference to an instance of a function local static
   * `domain` object.
   *
   * Uses the "construct on first use" idiom to safely ensure the `domain`
   * object is initialized exactly once upon first invocation of
   * `domain::get<DomainName>()`. All following invocations will return a
   * reference to the previously constructed `domain` object. See
   * https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
   *
   * None of the constructs in this header require the user to directly invoke
   * `domain::get`. It is automatically invoked when constructing objects like
   * a `domain_thread_range` or `category`. Advanced users may wish to use
   * `domain::get` for the convenience of the "construct on first use" idiom
   * when using domains with their own use of the NVTX C API.
   *
   * This function is threadsafe as of C++11. If two or more threads call
   * `domain::get<DomainName>` concurrently, exactly one of them is guaranteed
   * to construct the `domain` object and the other(s) will receive a
   * reference to the object after it is fully constructed.
   *
   * The domain's name is specified via the type `DomainName` pass as an
   * explicit template parameter. `DomainName` is required to contain a
   * member `DomainName::name` that resolves to either a `char const*` or
   * `wchar_t const*`. The value of `DomainName::name` is used to name and
   * uniquely identify the `domain`.
   *
   * Example:
   * ```
   * // The type `my_domain` defines a `name` member used to name and identify
   * // the `domain` object identified by `my_domain`.
   * struct my_domain{ static constexpr char const* name{"my domain"}; };
   *
   * auto D = domain::get<my_domain>(); // First invocation constructs a
   *                                    // `domain` with the name "my domain"
   *
   * auto D1 = domain::get<my_domain>(); // Simply returns reference to
   *                                     // previously constructed `domain`.
   * ```
   *
   * @tparam DomainName Type that contains a `DomainName::name` member used to
   * name the `domain` object.
   * @return Reference to the `domain` corresponding to the type `DomainName`.
   */
  template <typename DomainName>
  static domain const& get()
  {
    static_assert(detail::has_name_member<DomainName>(),
                  "Type used to identify a domain must contain a name member of"
                  "type const char* or const wchar_t*");
    static domain const d{DomainName::name};
    return d;
  }

  /**
   * @brief Conversion operator to `nvtxDomainHandle_t`.
   *
   * Allows transparently passing a domain object into an API expecting a
   * native `nvtxDomainHandle_t` object.
   */
  operator nvtxDomainHandle_t() const noexcept { return _domain; }

  /**
   * @brief Tag type for the "global" NVTX domain.
   *
   * This type may be passed as a template argument to any function/class
   * expecting a type to identify a domain to indicate that the global domain
   * should be used.
   *
   * All NVTX events in the global domain across all libraries and
   * applications will be grouped together.
   *
   */
  struct global {
  };

 private:
  /**
   * @brief Construct a new domain with the specified `name`.
   *
   * This constructor is private as it is intended that `domain` objects only
   * be created through the `domain::get` function.
   *
   * @param name A unique name identifying the domain
   */
  explicit domain(char const* name) noexcept : _domain{nvtxDomainCreateA(name)} {}

  /**
   * @brief Construct a new domain with the specified `name`.
   *
   * This constructor is private as it is intended that `domain` objects only
   * be created through the `domain::get` function.
   *
   * @param name A unique name identifying the domain
   */
  explicit domain(wchar_t const* name) noexcept : _domain{nvtxDomainCreateW(name)} {}

  /**
   * @brief Construct a new domain with the specified `name`.
   *
   * This constructor is private as it is intended that `domain` objects only
   * be created through the `domain::get` function.
   *
   * @param name A unique name identifying the domain
   */
  explicit domain(std::string const& name) noexcept : domain{name.c_str()} {}

  /**
   * @brief Construct a new domain with the specified `name`.
   *
   * This constructor is private as it is intended that `domain` objects only
   * be created through the `domain::get` function.
   *
   * @param name A unique name identifying the domain
   */
  explicit domain(std::wstring const& name) noexcept : domain{name.c_str()} {}

  /**
   * @brief Default constructor creates a `domain` representing the
   * "global" NVTX domain.
   *
   * All events not associated with a custom `domain` are grouped in the
   * "global" NVTX domain.
   *
   */
  domain() = default;

  /**
   * @brief Destroy the domain object, unregistering and freeing all domain
   * specific resources.
   */
  ~domain() noexcept { nvtxDomainDestroy(_domain); }

 private:
  nvtxDomainHandle_t const _domain{};  ///< The `domain`s NVTX handle
};

/**
 * @brief Returns reference to the `domain` object that represents the global
 * NVTX domain.
 *
 * This specialization for `domain::global` returns a default constructed,
 * `domain` object for use when the "global" domain is desired.
 *
 * All NVTX events in the global domain across all libraries and applications
 * will be grouped together.
 *
 * @return Reference to the `domain` corresponding to the global NVTX domain.
 *
 */
template <>
inline domain const& domain::get<domain::global>()
{
  static domain const d{};
  return d;
}

/**
 * @brief Indicates the values of the red, green, blue color channels for
 * a rgb color code.
 *
 */
struct rgb {
  /// Type used for component values
  using component_type = uint8_t;

  /**
   * @brief Construct a rgb with red, green, and blue channels
   * specified by `red_`, `green_`, and `blue_`, respectively.
   *
   * Valid values are in the range `[0,255]`.
   *
   * @param red_ Value of the red channel
   * @param green_ Value of the green channel
   * @param blue_ Value of the blue channel
   */
  constexpr rgb(component_type red_, component_type green_, component_type blue_) noexcept
    : red{red_}, green{green_}, blue{blue_}
  {
  }

  component_type const red{};    ///< Red channel value
  component_type const green{};  ///< Green channel value
  component_type const blue{};   ///< Blue channel value
};

/**
 * @brief Indicates the value of the alpha, red, green, and blue color
 * channels for an argb color code.
 *
 */
struct argb final : rgb {
  /**
   * @brief Construct an argb with alpha, red, green, and blue channels
   * specified by `alpha_`, `red_`, `green_`, and `blue_`, respectively.
   *
   * Valid values are in the range `[0,255]`.
   *
   * @param alpha_  Value of the alpha channel (opacity)
   * @param red_  Value of the red channel
   * @param green_  Value of the green channel
   * @param blue_  Value of the blue channel
   *
   */
  constexpr argb(component_type alpha_,
                 component_type red_,
                 component_type green_,
                 component_type blue_) noexcept
    : rgb{red_, green_, blue_}, alpha{alpha_}
  {
  }

  component_type const alpha{};  ///< Alpha channel value
};

/**
 * @brief Represents a custom color that can be associated with an NVTX event
 * via it's `event_attributes`.
 *
 * Specifying colors for NVTX events is a convenient way to visually
 * differentiate among different events in a visualization tool such as Nsight
 * Systems.
 *
 */
class color {
 public:
  /// Type used for the color's value
  using value_type = uint32_t;

  /**
   * @brief Constructs a `color` using the value provided by `hex_code`.
   *
   * `hex_code` is expected to be a 4 byte argb hex code.
   *
   * The most significant byte indicates the value of the alpha channel
   * (opacity) (0-255)
   *
   * The next byte indicates the value of the red channel (0-255)
   *
   * The next byte indicates the value of the green channel (0-255)
   *
   * The least significant byte indicates the value of the blue channel
   * (0-255)
   *
   * @param hex_code The hex code used to construct the `color`
   */
  constexpr explicit color(value_type hex_code) noexcept : _value{hex_code} {}

  /**
   * @brief Construct a `color` using the alpha, red, green, blue components
   * in `argb`.
   *
   * @param argb The alpha, red, green, blue components of the desired `color`
   */
  constexpr color(argb argb) noexcept
    : color{from_bytes_msb_to_lsb(argb.alpha, argb.red, argb.green, argb.blue)}
  {
  }

  /**
   * @brief Construct a `color` using the red, green, blue components in
   * `rgb`.
   *
   * Uses maximum value for the alpha channel (opacity) of the `color`.
   *
   * @param rgb The red, green, blue components of the desired `color`
   */
  constexpr color(rgb rgb) noexcept
    : color{from_bytes_msb_to_lsb(0xFF, rgb.red, rgb.green, rgb.blue)}
  {
  }

  /**
   * @brief Returns the `color`s argb hex code
   *
   */
  constexpr value_type get_value() const noexcept { return _value; }

  /**
   * @brief Return the NVTX color type of the color.
   *
   */
  constexpr nvtxColorType_t get_type() const noexcept { return _type; }

  color()             = delete;
  ~color()            = default;
  color(color const&) = default;
  color& operator=(color const&) = default;
  color(color&&)                 = default;
  color& operator=(color&&) = default;

 private:
  /**
   * @brief Constructs an unsigned, 4B integer from the component bytes in
   * most to least significant byte order.
   *
   */
  constexpr static value_type from_bytes_msb_to_lsb(uint8_t byte3,
                                                    uint8_t byte2,
                                                    uint8_t byte1,
                                                    uint8_t byte0) noexcept
  {
    return uint32_t{byte3} << 24 | uint32_t{byte2} << 16 | uint32_t{byte1} << 8 | uint32_t{byte0};
  }

  value_type const _value{};                     ///< color's argb color code
  nvtxColorType_t const _type{NVTX_COLOR_ARGB};  ///< NVTX color type code
};

/**
 * @brief Object for intra-domain grouping of NVTX events.
 *
 * A `category` is simply an integer id that allows for fine-grain grouping of
 * NVTX events. For example, one might use separate categories for IO, memory
 * allocation, compute, etc.
 *
 * Example:
 * \code{.cpp}
 * nvtx3::category cat1{1};
 *
 * // Range `r1` belongs to the category identified by the value `1`.
 * nvtx3::thread_range r1{cat1};
 *
 * // Range `r2` belongs to the same category as `r1`
 * nvtx3::thread_range r2{nvtx3::category{1}};
 * \endcode
 *
 * To associate a name string with a category id, see `named_category`.
 *
 */
class category {
 public:
  /// Type used for `category`s integer id.
  using id_type = uint32_t;

  /**
   * @brief Construct a `category` with the specified `id`.
   *
   * The `category` will be unnamed and identified only by its `id` value.
   *
   * All `category` objects sharing the same `id` are equivalent.
   *
   * @param[in] id The `category`'s identifying value
   */
  constexpr explicit category(id_type id) noexcept : id_{id} {}

  /**
   * @brief Returns the id of the category.
   *
   */
  constexpr id_type get_id() const noexcept { return id_; }

  category()                = delete;
  ~category()               = default;
  category(category const&) = default;
  category& operator=(category const&) = default;
  category(category&&)                 = default;
  category& operator=(category&&) = default;

 private:
  id_type const id_{};  ///< category's unique identifier
};

/**
 * @brief A `category` with an associated name string.
 *
 * Associates a `name` string with a category `id` to help differentiate among
 * categories.
 *
 * For any given category id `Id`, a `named_category(Id, "name")` should only
 * be constructed once and reused throughout an application. This can be done
 * by either explicitly creating static `named_category` objects, or using the
 * `named_category::get` construct on first use helper (recommended).
 *
 * Creating two or more `named_category` objects with the same value for `id`
 * in the same domain results in undefined behavior.
 *
 * Similarly, behavior is undefined when a `named_category` and `category`
 * share the same value of `id`.
 *
 * Example:
 * \code{.cpp}
 * // Explicitly constructed, static `named_category`
 * static nvtx3::named_category static_category{42, "my category"};
 *
 * // Range `r` associated with category id `42`
 * nvtx3::thread_range r{static_category};
 *
 * // OR use construct on first use:
 *
 * // Define a type with `name` and `id` members
 * struct my_category{
 *    static constexpr char const* name{"my category"}; // category name
 *    static constexpr category::id_type id{42}; // category id
 * };
 *
 * // Use construct on first use to name the category id `42`
 * // with name "my category"
 * auto my_category = named_category<my_domain>::get<my_category>();
 *
 * // Range `r` associated with category id `42`
 * nvtx3::thread_range r{my_category};
 * \endcode
 *
 * `named_category`'s association of a name to a category id is local to the
 * domain specified by the type `D`. An id may have a different name in
 * another domain.
 *
 * @tparam D Type containing `name` member used to identify the `domain` to
 * which the `named_category` belongs. Else, `domain::global` to  indicate
 * that the global NVTX domain should be used.
 */
template <typename D = domain::global>
class named_category final : public category {
 public:
  /**
   * @brief Returns a global instance of a `named_category` as a
   * function-local static.
   *
   * Creates a `named_category` with name and id specified by the contents of
   * a type `C`. `C::name` determines the name and `C::id` determines the
   * category id.
   *
   * This function is useful for constructing a named `category` exactly once
   * and reusing the same instance throughout an application.
   *
   * Example:
   * \code{.cpp}
   * // Define a type with `name` and `id` members
   * struct my_category{
   *    static constexpr char const* name{"my category"}; // category name
   *    static constexpr uint32_t id{42}; // category id
   * };
   *
   * // Use construct on first use to name the category id `42`
   * // with name "my category"
   * auto cat = named_category<my_domain>::get<my_category>();
   *
   * // Range `r` associated with category id `42`
   * nvtx3::thread_range r{cat};
   * \endcode
   *
   * Uses the "construct on first use" idiom to safely ensure the `category`
   * object is initialized exactly once. See
   * https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
   *
   * @tparam C Type containing a member `C::name` that resolves  to either a
   * `char const*` or `wchar_t const*` and `C::id`.
   */
  template <typename C>
  static named_category<D> const& get() noexcept
  {
    static_assert(detail::has_name_member<C>(),
                  "Type used to name a category must contain a name member.");
    static named_category<D> const category{C::id, C::name};
    return category;
  }
  /**
   * @brief Construct a `category` with the specified `id` and `name`.
   *
   * The name `name` will be registered with `id`.
   *
   * Every unique value of `id` should only be named once.
   *
   * @param[in] id The category id to name
   * @param[in] name The name to associated with `id`
   */
  named_category(id_type id, char const* name) noexcept : category{id}
  {
    nvtxDomainNameCategoryA(domain::get<D>(), get_id(), name);
  };

  /**
   * @brief Construct a `category` with the specified `id` and `name`.
   *
   * The name `name` will be registered with `id`.
   *
   * Every unique value of `id` should only be named once.
   *
   * @param[in] id The category id to name
   * @param[in] name The name to associated with `id`
   */
  named_category(id_type id, wchar_t const* name) noexcept : category{id}
  {
    nvtxDomainNameCategoryW(domain::get<D>(), get_id(), name);
  };
};

/**
 * @brief A message registered with NVTX.
 *
 * Normally, associating a `message` with an NVTX event requires copying the
 * contents of the message string. This may cause non-trivial overhead in
 * highly performance sensitive regions of code.
 *
 * message registration is an optimization to lower the overhead of
 * associating a message with an NVTX event. Registering a message yields a
 * handle that is inexpensive to copy that may be used in place of a message
 * string.
 *
 * A particular message should only be registered once and the handle
 * reused throughout the rest of the application. This can be done by either
 * explicitly creating static `registered_message` objects, or using the
 * `registered_message::get` construct on first use helper (recommended).
 *
 * Example:
 * \code{.cpp}
 * // Explicitly constructed, static `registered_message`
 * static registered_message<my_domain> static_message{"message"};
 *
 * // "message" is associated with the range `r`
 * nvtx3::thread_range r{static_message};
 *
 * // Or use construct on first use:
 *
 * // Define a type with a `message` member that defines the contents of the
 * // registered message
 * struct my_message{ static constexpr char const* message{ "my message" }; };
 *
 * // Uses construct on first use to register the contents of
 * // `my_message::message`
 * auto msg = registered_message<my_domain>::get<my_message>();
 *
 * // "my message" is associated with the range `r`
 * nvtx3::thread_range r{msg};
 * \endcode
 *
 * `registered_message`s are local to a particular domain specified via
 * the type `D`.
 *
 * @tparam D Type containing `name` member used to identify the `domain` to
 * which the `registered_message` belongs. Else, `domain::global` to  indicate
 * that the global NVTX domain should be used.
 */
template <typename D = domain::global>
class registered_message {
 public:
  /**
   * @brief Returns a global instance of a `registered_message` as a function
   * local static.
   *
   * Provides a convenient way to register a message with NVTX without having
   * to explicitly register the message.
   *
   * Upon first invocation, constructs a `registered_message` whose contents
   * are specified by `message::message`.
   *
   * All future invocations will return a reference to the object constructed
   * in the first invocation.
   *
   * Example:
   * \code{.cpp}
   * // Define a type with a `message` member that defines the contents of the
   * // registered message
   * struct my_message{ static constexpr char const* message{ "my message" };
   * };
   *
   * // Uses construct on first use to register the contents of
   * // `my_message::message`
   * auto msg = registered_message<my_domain>::get<my_message>();
   *
   * // "my message" is associated with the range `r`
   * nvtx3::thread_range r{msg};
   * \endcode
   *
   * @tparam M Type required to contain a member `M::message` that
   * resolves to either a `char const*` or `wchar_t const*` used as the
   * registered message's contents.
   * @return Reference to a `registered_message` associated with the type `M`.
   */
  template <typename M>
  static registered_message<D> const& get() noexcept
  {
    static registered_message<D> const registered_message{M::message};
    return registered_message;
  }

  /**
   * @brief Constructs a `registered_message` from the specified `msg` string.
   *
   * Registers `msg` with NVTX and associates a handle with the registered
   * message.
   *
   * A particular message should should only be registered once and the handle
   * reused throughout the rest of the application.
   *
   * @param msg The contents of the message
   */
  explicit registered_message(char const* msg) noexcept
    : handle_{nvtxDomainRegisterStringA(domain::get<D>(), msg)}
  {
  }

  /**
   * @brief Constructs a `registered_message` from the specified `msg` string.
   *
   * Registers `msg` with NVTX and associates a handle with the registered
   * message.
   *
   * A particular message should should only be registered once and the handle
   * reused throughout the rest of the application.
   *
   * @param msg The contents of the message
   */
  explicit registered_message(std::string const& msg) noexcept : registered_message{msg.c_str()} {}

  /**
   * @brief Constructs a `registered_message` from the specified `msg` string.
   *
   * Registers `msg` with NVTX and associates a handle with the registered
   * message.
   *
   * A particular message should should only be registered once and the handle
   * reused throughout the rest of the application.
   *
   * @param msg The contents of the message
   */
  explicit registered_message(wchar_t const* msg) noexcept
    : handle_{nvtxDomainRegisterStringW(domain::get<D>(), msg)}
  {
  }

  /**
   * @brief Constructs a `registered_message` from the specified `msg` string.
   *
   * Registers `msg` with NVTX and associates a handle with the registered
   * message.
   *
   * A particular message should only be registered once and the handle
   * reused throughout the rest of the application.
   *
   * @param msg The contents of the message
   */
  explicit registered_message(std::wstring const& msg) noexcept : registered_message{msg.c_str()} {}

  /**
   * @brief Returns the registered message's handle
   *
   */
  nvtxStringHandle_t get_handle() const noexcept { return handle_; }

  registered_message()                          = delete;
  ~registered_message()                         = default;
  registered_message(registered_message const&) = default;
  registered_message& operator=(registered_message const&) = default;
  registered_message(registered_message&&)                 = default;
  registered_message& operator=(registered_message&&) = default;

 private:
  nvtxStringHandle_t const handle_{};  ///< The handle returned from
                                       ///< registering the message with NVTX
};

/**
 * @brief Allows associating a message string with an NVTX event via
 * its `EventAttribute`s.
 *
 * Associating a `message` with an NVTX event through its `event_attributes`
 * allows for naming events to easily differentiate them from other events.
 *
 * Every time an NVTX event is created with an associated `message`, the
 * contents of the message string must be copied.  This may cause non-trivial
 * overhead in highly performance sensitive sections of code. Use of a
 * `nvtx3::registered_message` is recommended in these situations.
 *
 * Example:
 * \code{.cpp}
 * // Creates an `event_attributes` with message "message 0"
 * nvtx3::event_attributes attr0{nvtx3::message{"message 0"}};
 *
 * // `range0` contains message "message 0"
 * nvtx3::thread_range range0{attr0};
 *
 * // `std::string` and string literals are implicitly assumed to be
 * // the contents of an `nvtx3::message`
 * // Creates an `event_attributes` with message "message 1"
 * nvtx3::event_attributes attr1{"message 1"};
 *
 * // `range1` contains message "message 1"
 * nvtx3::thread_range range1{attr1};
 *
 * // `range2` contains message "message 2"
 * nvtx3::thread_range range2{nvtx3::Mesage{"message 2"}};
 *
 * // `std::string` and string literals are implicitly assumed to be
 * // the contents of an `nvtx3::message`
 * // `range3` contains message "message 3"
 * nvtx3::thread_range range3{"message 3"};
 * \endcode
 */
class message {
 public:
  using value_type = nvtxMessageValue_t;

  /**
   * @brief Construct a `message` whose contents are specified by `msg`.
   *
   * @param msg The contents of the message
   */
  NVTX3_RELAXED_CONSTEXPR message(char const* msg) noexcept : type_{NVTX_MESSAGE_TYPE_ASCII}
  {
    value_.ascii = msg;
  }

  /**
   * @brief Construct a `message` whose contents are specified by `msg`.
   *
   * @param msg The contents of the message
   */
  message(std::string const& msg) noexcept : message{msg.c_str()} {}

  /**
   * @brief Disallow construction for `std::string` r-value
   *
   * `message` is a non-owning type and therefore cannot take ownership of an
   * r-value. Therefore, constructing from an r-value is disallowed to prevent
   * a dangling pointer.
   *
   */
  message(std::string&&) = delete;

  /**
   * @brief Construct a `message` whose contents are specified by `msg`.
   *
   * @param msg The contents of the message
   */
  NVTX3_RELAXED_CONSTEXPR message(wchar_t const* msg) noexcept : type_{NVTX_MESSAGE_TYPE_UNICODE}
  {
    value_.unicode = msg;
  }

  /**
   * @brief Construct a `message` whose contents are specified by `msg`.
   *
   * @param msg The contents of the message
   */
  message(std::wstring const& msg) noexcept : message{msg.c_str()} {}

  /**
   * @brief Disallow construction for `std::wstring` r-value
   *
   * `message` is a non-owning type and therefore cannot take ownership of an
   * r-value. Therefore, constructing from an r-value is disallowed to prevent
   * a dangling pointer.
   *
   */
  message(std::wstring&&) = delete;

  /**
   * @brief Construct a `message` from a `registered_message`.
   *
   * @tparam D Type containing `name` member used to identify the `domain`
   * to which the `registered_message` belongs. Else, `domain::global` to
   * indicate that the global NVTX domain should be used.
   * @param msg The message that has already been registered with NVTX.
   */
  template <typename D>
  message(registered_message<D> const& msg) noexcept : type_{NVTX_MESSAGE_TYPE_REGISTERED}
  {
    value_.registered = msg.get_handle();
  }

  /**
   * @brief Return the union holding the value of the message.
   *
   */
  NVTX3_RELAXED_CONSTEXPR value_type get_value() const noexcept { return value_; }

  /**
   * @brief Return the type information about the value the union holds.
   *
   */
  NVTX3_RELAXED_CONSTEXPR nvtxMessageType_t get_type() const noexcept { return type_; }

 private:
  nvtxMessageType_t const type_{};  ///< message type
  nvtxMessageValue_t value_{};      ///< message contents
};

/**
 * @brief A numerical value that can be associated with an NVTX event via
 * its `event_attributes`.
 *
 * Example:
 * ```
 * nvtx3:: event_attributes attr{nvtx3::payload{42}}; // Constructs a payload
 * from
 *                                                 // the `int32_t` value 42
 *
 * // `range0` will have an int32_t payload of 42
 * nvtx3::thread_range range0{attr};
 *
 * // range1 has double payload of 3.14
 * nvtx3::thread_range range1{ nvtx3::payload{3.14} };
 * ```
 */
class payload {
 public:
  using value_type = typename nvtxEventAttributes_v2::payload_t;

  /**
   * @brief Construct a `payload` from a signed, 8 byte integer.
   *
   * @param value Value to use as contents of the payload
   */
  NVTX3_RELAXED_CONSTEXPR explicit payload(int64_t value) noexcept
    : type_{NVTX_PAYLOAD_TYPE_INT64}, value_{}
  {
    value_.llValue = value;
  }

  /**
   * @brief Construct a `payload` from a signed, 4 byte integer.
   *
   * @param value Value to use as contents of the payload
   */
  NVTX3_RELAXED_CONSTEXPR explicit payload(int32_t value) noexcept
    : type_{NVTX_PAYLOAD_TYPE_INT32}, value_{}
  {
    value_.iValue = value;
  }

  /**
   * @brief Construct a `payload` from an unsigned, 8 byte integer.
   *
   * @param value Value to use as contents of the payload
   */
  NVTX3_RELAXED_CONSTEXPR explicit payload(uint64_t value) noexcept
    : type_{NVTX_PAYLOAD_TYPE_UNSIGNED_INT64}, value_{}
  {
    value_.ullValue = value;
  }

  /**
   * @brief Construct a `payload` from an unsigned, 4 byte integer.
   *
   * @param value Value to use as contents of the payload
   */
  NVTX3_RELAXED_CONSTEXPR explicit payload(uint32_t value) noexcept
    : type_{NVTX_PAYLOAD_TYPE_UNSIGNED_INT32}, value_{}
  {
    value_.uiValue = value;
  }

  /**
   * @brief Construct a `payload` from a single-precision floating point
   * value.
   *
   * @param value Value to use as contents of the payload
   */
  NVTX3_RELAXED_CONSTEXPR explicit payload(float value) noexcept
    : type_{NVTX_PAYLOAD_TYPE_FLOAT}, value_{}
  {
    value_.fValue = value;
  }

  /**
   * @brief Construct a `payload` from a double-precision floating point
   * value.
   *
   * @param value Value to use as contents of the payload
   */
  NVTX3_RELAXED_CONSTEXPR explicit payload(double value) noexcept
    : type_{NVTX_PAYLOAD_TYPE_DOUBLE}, value_{}
  {
    value_.dValue = value;
  }

  /**
   * @brief Return the union holding the value of the payload
   *
   */
  NVTX3_RELAXED_CONSTEXPR value_type get_value() const noexcept { return value_; }

  /**
   * @brief Return the information about the type the union holds.
   *
   */
  NVTX3_RELAXED_CONSTEXPR nvtxPayloadType_t get_type() const noexcept { return type_; }

 private:
  nvtxPayloadType_t const type_;  ///< Type of the payload value
  value_type value_;              ///< Union holding the payload value
};

/**
 * @brief Describes the attributes of a NVTX event.
 *
 * NVTX events can be customized via four "attributes":
 *
 * - color:    color used to visualize the event in tools such as Nsight
 *             Systems. See `color`.
 * - message:  Custom message string. See `message`.
 * - payload:  User-defined numerical value. See `payload`.
 * - category: Intra-domain grouping. See `category`.
 *
 * These component attributes are specified via an `event_attributes` object.
 * See `nvtx3::color`, `nvtx3::message`, `nvtx3::payload`, and
 * `nvtx3::category` for how these individual attributes are constructed.
 *
 * While it is possible to specify all four attributes, it is common to want
 * to only specify a subset of attributes and use default values for the
 * others. For convenience, `event_attributes` can be constructed from any
 * number of attribute components in any order.
 *
 * Example:
 * \code{.cpp}
 * event_attributes attr{}; // No arguments, use defaults for all attributes
 *
 * event_attributes attr{"message"}; // Custom message, rest defaulted
 *
 * // Custom color & message
 * event_attributes attr{"message", nvtx3::rgb{127, 255, 0}};
 *
 * /// Custom color & message, can use any order of arguments
 * event_attributes attr{nvtx3::rgb{127, 255, 0}, "message"};
 *
 *
 * // Custom color, message, payload, category
 * event_attributes attr{nvtx3::rgb{127, 255, 0},
 *                      "message",
 *                      nvtx3::payload{42},
 *                      nvtx3::category{1}};
 *
 * // Custom color, message, payload, category, can use any order of arguments
 * event_attributes attr{nvtx3::payload{42},
 *                      nvtx3::category{1},
 *                      "message",
 *                      nvtx3::rgb{127, 255, 0}};
 *
 * // Multiple arguments of the same type are allowed, but only the first is
 * // used. All others are ignored
 * event_attributes attr{ nvtx3::payload{42}, nvtx3::payload{7} }; // payload
 * is 42
 *
 * // Range `r` will be customized according the attributes in `attr`
 * nvtx3::thread_range r{attr};
 *
 * // For convenience, the arguments that can be passed to the
 * `event_attributes`
 * // constructor may be passed to the `domain_thread_range` contructor where
 * // they will be forwarded to the `EventAttribute`s constructor
 * nvtx3::thread_range r{nvtx3::payload{42}, nvtx3::category{1}, "message"};
 * \endcode
 *
 */
class event_attributes {
 public:
  using value_type = nvtxEventAttributes_t;

  /**
   * @brief Default constructor creates an `event_attributes` with no
   * category, color, payload, nor message.
   */
  constexpr event_attributes() noexcept
    : attributes_{
        NVTX_VERSION,                   // version
        sizeof(nvtxEventAttributes_t),  // size
        0,                              // category
        NVTX_COLOR_UNKNOWN,             // color type
        0,                              // color value
        NVTX_PAYLOAD_UNKNOWN,           // payload type
        {},                             // payload value (union)
        NVTX_MESSAGE_UNKNOWN,           // message type
        {}                              // message value (union)
      }
  {
  }

  /**
   * @brief Variadic constructor where the first argument is a `category`.
   *
   * Sets the value of the `EventAttribute`s category based on `c` and
   * forwards the remaining variadic parameter pack to the next constructor.
   *
   */
  template <typename... Args>
  NVTX3_RELAXED_CONSTEXPR explicit event_attributes(category const& c, Args const&... args) noexcept
    : event_attributes(args...)
  {
    attributes_.category = c.get_id();
  }

  /**
   * @brief Variadic constructor where the first argument is a `color`.
   *
   * Sets the value of the `EventAttribute`s color based on `c` and forwards
   * the remaining variadic parameter pack to the next constructor.
   *
   */
  template <typename... Args>
  NVTX3_RELAXED_CONSTEXPR explicit event_attributes(color const& c, Args const&... args) noexcept
    : event_attributes(args...)
  {
    attributes_.color     = c.get_value();
    attributes_.colorType = c.get_type();
  }

  /**
   * @brief Variadic constructor where the first argument is a `payload`.
   *
   * Sets the value of the `EventAttribute`s payload based on `p` and forwards
   * the remaining variadic parameter pack to the next constructor.
   *
   */
  template <typename... Args>
  NVTX3_RELAXED_CONSTEXPR explicit event_attributes(payload const& p, Args const&... args) noexcept
    : event_attributes(args...)
  {
    attributes_.payload     = p.get_value();
    attributes_.payloadType = p.get_type();
  }

  /**
   * @brief Variadic constructor where the first argument is a `message`.
   *
   * Sets the value of the `EventAttribute`s message based on `m` and forwards
   * the remaining variadic parameter pack to the next constructor.
   *
   */
  template <typename... Args>
  explicit event_attributes(message const& m, Args const&... args) noexcept
    : event_attributes(args...)
  {
    attributes_.message     = m.get_value();
    attributes_.messageType = m.get_type();
  }

  ~event_attributes()                       = default;
  event_attributes(event_attributes const&) = default;
  event_attributes& operator=(event_attributes const&) = default;
  event_attributes(event_attributes&&)                 = default;
  event_attributes& operator=(event_attributes&&) = default;

  /**
   * @brief Get raw pointer to underlying NVTX attributes object.
   *
   */
  constexpr value_type const* get() const noexcept { return &attributes_; }

 private:
  value_type attributes_{};  ///< The NVTX attributes structure
};

/**
 * @brief A RAII object for creating a NVTX range local to a thread within a
 * domain.
 *
 * When constructed, begins a nested NVTX range on the calling thread in the
 * specified domain. Upon destruction, ends the NVTX range.
 *
 * Behavior is undefined if a `domain_thread_range` object is
 * created/destroyed on different threads.
 *
 * `domain_thread_range` is neither moveable nor copyable.
 *
 * `domain_thread_range`s may be nested within other ranges.
 *
 * The domain of the range is specified by the template type parameter `D`.
 * By default, the `domain::global` is used, which scopes the range to the
 * global NVTX domain. The convenience alias `thread_range` is provided for
 * ranges scoped to the global domain.
 *
 * A custom domain can be defined by creating a type, `D`, with a static
 * member `D::name` whose value is used to name the domain associated with
 * `D`. `D::name` must resolve to either `char const*` or `wchar_t const*`
 *
 * Example:
 * ```
 * // Define a type `my_domain` with a member `name` used to name the domain
 * // associated with the type `my_domain`.
 * struct my_domain{
 *    static constexpr const char * name{"my domain"};
 * };
 * ```
 *
 * Usage:
 * ```
 * nvtx3::domain_thread_range<> r0{"range 0"}; // Range in global domain
 *
 * nvtx3::thread_range r1{"range 1"}; // Alias for range in global domain
 *
 * nvtx3::domain_thread_range<my_domain> r2{"range 2"}; // Range in custom
 * domain
 *
 * // specify an alias to a range that uses a custom domain
 * using my_thread_range = nvtx3::domain_thread_range<my_domain>;
 *
 * my_thread_range r3{"range 3"}; // Alias for range in custom domain
 * ```
 */
template <class D = domain::global>
class domain_thread_range {
 public:
  /**
   * @brief Construct a `domain_thread_range` with the specified
   * `event_attributes`
   *
   * Example:
   * ```
   * nvtx3::event_attributes attr{"msg", nvtx3::rgb{127,255,0}};
   * nvtx3::domain_thread_range<> range{attr}; // Creates a range with message
   * contents
   *                                    // "msg" and green color
   * ```
   *
   * @param[in] attr `event_attributes` that describes the desired attributes
   * of the range.
   */
  explicit domain_thread_range(event_attributes const& attr) noexcept
  {
    nvtxDomainRangePushEx(domain::get<D>(), attr.get());
  }

  /**
   * @brief Constructs a `domain_thread_range` from the constructor arguments
   * of an `event_attributes`.
   *
   * Forwards the arguments `first, args...` to construct an
   * `event_attributes` object. The `event_attributes` object is then
   * associated with the `domain_thread_range`.
   *
   * For more detail, see `event_attributes` documentation.
   *
   * Example:
   * ```
   * // Creates a range with message "message" and green color
   * nvtx3::domain_thread_range<> r{"message", nvtx3::rgb{127,255,0}};
   * ```
   *
   * @note To prevent making needless copies of `event_attributes` objects,
   * this constructor is disabled when the first argument is an
   * `event_attributes` object, instead preferring the explicit
   * `domain_thread_range(event_attributes const&)` constructor.
   *
   * @param[in] first First argument to forward to the `event_attributes`
   * constructor.
   * @param[in] args Variadic parameter pack of additional arguments to
   * forward.
   *
   */
  template <typename First,
            typename... Args,
            typename = typename std::enable_if<
              not std::is_same<event_attributes, typename std::decay<First>>::value>>
  explicit domain_thread_range(First const& first, Args const&... args) noexcept
    : domain_thread_range{event_attributes{first, args...}}
  {
  }

  /**
   * @brief Default constructor creates a `domain_thread_range` with no
   * message, color, payload, nor category.
   *
   */
  domain_thread_range() : domain_thread_range{event_attributes{}} {}

  domain_thread_range(domain_thread_range const&) = delete;
  domain_thread_range& operator=(domain_thread_range const&) = delete;
  domain_thread_range(domain_thread_range&&)                 = delete;
  domain_thread_range& operator=(domain_thread_range&&) = delete;

  /**
   * @brief Destroy the domain_thread_range, ending the NVTX range event.
   */
  ~domain_thread_range() noexcept { nvtxDomainRangePop(domain::get<D>()); }
};

/**
 * @brief Alias for a `domain_thread_range` in the global NVTX domain.
 *
 */
using thread_range = domain_thread_range<>;

/**
 * @brief A RAII object for creating a NVTX range within a domain that can be
 * created and destroyed on different threads.
 *
 * When constructed, begins a NVTX range in the specified domain. Upon
 * destruction, ends the NVTX range.
 *
 * Similar to `nvtx3::domain_thread_range`, the only difference being that
 * `domain_process_range` can start and end on different threads.
 *
 * Use of `nvtx3::domain_thread_range` should be preferred unless one needs
 * the ability to start and end a range on different threads.
 *
 * `domain_process_range` is moveable, but not copyable.
 *
 * @tparam D Type containing `name` member used to identify the `domain`
 * to which the `domain_process_range` belongs. Else, `domain::global` to
 * indicate that the global NVTX domain should be used.
 */
template <typename D = domain::global>
class domain_process_range {
 public:
  /**
   * @brief Construct a new domain process range object
   *
   * @param attr
   */
  explicit domain_process_range(event_attributes const& attr) noexcept
    : range_id_{nvtxDomainRangeStartEx(domain::get<D>(), attr.get())}
  {
  }

  /**
   * @brief Construct a new domain process range object
   *
   * @param first
   * @param args
   */
  template <typename First,
            typename... Args,
            typename = typename std::enable_if<
              not std::is_same<event_attributes, typename std::decay<First>>::value>>
  explicit domain_process_range(First const& first, Args const&... args) noexcept
    : domain_process_range{event_attributes{first, args...}}
  {
  }

  /**
   * @brief Construct a new domain process range object
   *
   */
  constexpr domain_process_range() noexcept : domain_process_range{event_attributes{}} {}

  /**
   * @brief Destroy the `domain_process_range` ending the range.
   *
   */
  ~domain_process_range() noexcept
  {
    if (not moved_from_) { nvtxRangeEnd(range_id_); }
  }

  domain_process_range(domain_process_range const&) = delete;
  domain_process_range& operator=(domain_process_range const&) = delete;

  domain_process_range(domain_process_range&& other) noexcept : range_id_{other.range_id_}
  {
    other.moved_from_ = true;
  }

  domain_process_range& operator=(domain_process_range&& other) noexcept
  {
    range_id_         = other.range_id_;
    other.moved_from_ = true;
  }

 private:
  nvtxRangeId_t range_id_;  ///< Range id used to correlate
                            ///< the start/end of the range
  bool moved_from_{false};  ///< Indicates if the object has had
                            ///< it's contents moved from it,
                            ///< indicating it should not attempt
                            ///< to end the NVTX range.
};

/**
 * @brief Alias for a `domain_process_range` in the global NVTX domain.
 *
 */
using process_range = domain_process_range<>;

/**
 * @brief Annotates an instantaneous point in time with the attributes specified
 * by `attr`.
 *
 * Unlike a "range", a mark is an instantaneous event in an application, e.g.,
 * locking/unlocking a mutex.
 *
 * \code{.cpp}
 * std::mutex global_lock;
 * void lock_mutex(){
 *    global_lock.lock();
 *    nvtx3::mark("lock_mutex");
 * }
 * \endcode
 *
 * @tparam D Type containing `name` member used to identify the `domain`
 * to which the `domain_process_range` belongs. Else, `domain::global` to
 * indicate that the global NVTX domain should be used.
 * @param[in] attr `event_attributes` that describes the desired attributes
 * of the mark.
 */
template <typename D = nvtx3::domain::global>
inline void mark(event_attributes const& attr) noexcept
{
  nvtxDomainMarkEx(domain::get<D>(), attr.get());
}

}  // namespace nvtx3

/**
 * @brief Convenience macro for generating a range in the specified `domain`
 * from the lifetime of a function
 *
 * This macro is useful for generating an NVTX range in `domain` from
 * the entry point of a function to its exit. It is intended to be the first
 * line of the function.
 *
 * Constructs a static `registered_message` using the name of the immediately
 * enclosing function returned by `__func__` and constructs a
 * `nvtx3::thread_range` using the registered function name as the range's
 * message.
 *
 * Example:
 * ```
 * struct my_domain{static constexpr char const* name{"my_domain"};};
 *
 * void foo(...){
 *    NVTX3_FUNC_RANGE_IN(my_domain); // Range begins on entry to foo()
 *    // do stuff
 *    ...
 * } // Range ends on return from foo()
 * ```
 *
 * @param[in] D Type containing `name` member used to identify the
 * `domain` to which the `registered_message` belongs. Else,
 * `domain::global` to  indicate that the global NVTX domain should be used.
 */
#define NVTX3_FUNC_RANGE_IN(D)                                                 \
  static ::nvtx3::registered_message<D> const nvtx3_func_name__{__func__};     \
  static ::nvtx3::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
  ::nvtx3::domain_thread_range<D> const nvtx3_range__{nvtx3_func_attr__};

/**
 * @brief Convenience macro for generating a range in the global domain from the
 * lifetime of a function.
 *
 * This macro is useful for generating an NVTX range in the global domain from
 * the entry point of a function to its exit. It is intended to be the first
 * line of the function.
 *
 * Constructs a static `registered_message` using the name of the immediately
 * enclosing function returned by `__func__` and constructs a
 * `nvtx3::thread_range` using the registered function name as the range's
 * message.
 *
 * Example:
 * ```
 * void foo(...){
 *    NVTX3_FUNC_RANGE(); // Range begins on entry to foo()
 *    // do stuff
 *    ...
 * } // Range ends on return from foo()
 * ```
 */
#define NVTX3_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(::nvtx3::domain::global)
