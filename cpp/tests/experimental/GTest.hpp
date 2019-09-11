/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef NV_VPI_TEST_UTIL_GTEST_HPP
#define NV_VPI_TEST_UTIL_GTEST_HPP

#ifdef GTEST_INCLUDE_GTEST_GTEST_H_
#error "Don't include gtest/gtest.h directly, include util/GTest.hpp instead"
#endif

// I'm not proud of the following code, but we have to replace
// the hacky emulation of variadic templates for ::testing::Types
// for the real deal. The emulation is limited to 50 types, and
// we need way more than that.
//
// We'll just use macros to rename the emulated variadic template types
// and redefine them properly later. This way we don't have to directly
// modify gtest's header file.

#define Types NV_VPI_Types_NOT_USED
#define Types0 NV_VPI_Types0_NOT_USED
#define TypeList NV_VPI_TypeList_NOT_USED
#define Templates NV_VPI_Templates_NOT_USED
#define Templates0 NV_VPI_Templates0_NOT_USED
#include <gtest/internal/gtest-type-util.h>
#undef Types
#undef Types0
#undef TypeList
#undef Templates
#undef Templates0

#include <cmath> // for std::abs

namespace testing {

template<class... TYPES>
struct Types
{
    using type = Types;
};

template<class T, class... TYPES>
struct Types<T, TYPES...>
{
    using Head = T;
    using Tail = Types<TYPES...>;

    using type = Types;
};

namespace internal {

using Types0 = Types<>;

template<GTEST_TEMPLATE_... TYPES>
struct Templates
{
};

template<GTEST_TEMPLATE_ HEAD, GTEST_TEMPLATE_... TAIL>
struct Templates<HEAD, TAIL...>
{
    using Head = internal::TemplateSel<HEAD>;
    using Tail = Templates<TAIL...>;

    using type = Templates;
};

using Templates0 = Templates<>;

template<typename T>
struct TypeList
{
    typedef Types<T> type;
};

template<class... TYPES>
struct TypeList<Types<TYPES...>>
{
    using type = Types<TYPES...>;
};

} // namespace internal
} // namespace testing

#include <gtest/gtest.h>

#define VPI_EXPECT_PRIV_EXCEPTION(STMT, CODE)                                                                \
    try                                                                                                      \
    {                                                                                                        \
        STMT;                                                                                                \
        ADD_FAILURE() << "Should have thrown an exception";                                                  \
    }                                                                                                        \
    catch (::nv::vpi::priv::Exception & e)                                                                   \
    {                                                                                                        \
        EXPECT_EQ((CODE), e.code()) << "Should have thrown a nv::vpi::priv::Exception with code " << (CODE); \
    }                                                                                                        \
    catch (...)                                                                                              \
    {                                                                                                        \
        ADD_FAILURE() << "Should have thrown a nv::vpi::priv::Exception";                                    \
    }

#define VPI_ASSERT_PRIV_EXCEPTION(STMT, CODE)                                                                \
    try                                                                                                      \
    {                                                                                                        \
        STMT;                                                                                                \
        FAIL() << "Should have thrown an exception";                                                         \
    }                                                                                                        \
    catch (::nv::vpi::priv::Exception & e)                                                                   \
    {                                                                                                        \
        EXPECT_EQ((CODE), e.code()) << "Should have thrown a nv::vpi::priv::Exception with code " << (CODE); \
    }                                                                                                        \
    catch (...)                                                                                              \
    {                                                                                                        \
        FAIL() << "Should have thrown a nv::vpi::priv::Exception";                                           \
    }

#define VPI_ASSERT_THROW(E, ...)                                                                               \
    try                                                                                                        \
    {                                                                                                          \
        __VA_ARGS__;                                                                                           \
        ADD_FAILURE() << "Expected an exception of type " #E ", got none";                                     \
    }                                                                                                          \
    catch (E & e)                                                                                              \
    {                                                                                                          \
    }                                                                                                          \
    catch (std::exception & e)                                                                                 \
    {                                                                                                          \
        ADD_FAILURE() << "Expected an exception of type " #E ", got " << typeid(e).name() << " with message '" \
                      << e.what() << "'";                                                                      \
    }                                                                                                          \
    catch (...)                                                                                                \
    {                                                                                                          \
        ADD_FAILURE() << "Expected an exception of type " #E ", got an unknown exception";                     \
    }

namespace util {

template<class T>
void CheckArrayElementsAreNear(const T *expected, size_t sizeExpected, const T *test, size_t sizeTest, double threshold)
{
    ASSERT_EQ(sizeExpected, sizeTest);

    for (size_t i = 0; i < sizeExpected; ++i)
    {
        if (std::abs((double)expected[i] - (double)test[i]) > threshold)
        {
            goto fail; // Dijkstra is wrong in this case.
        }
    }
    return;

    namespace t = ::testing;

fail:
    std::cout << "Array has " << sizeExpected << " elements, they must differ by at most" << t::PrintToString(threshold)
              << '\n';

    for (size_t i = 0; i < sizeExpected; ++i)
    {
        std::cout << "element #" << i << ", " << t::PrintToString(test[i]);

        double diff = std::abs((double)expected[i] - (double)test[i]);

        if (diff == 0)
        {
            std::cout << ", is equal to " << t::PrintToString(expected[i]);
        }
        else if (diff > threshold)
        {
            std::cout << ", differs from " << t::PrintToString(expected[i]) << " by " << diff;
        }
        else
        {
            std::cout << ", is near " << t::PrintToString(expected[i]);
        }

        if (i < sizeExpected - 1)
        {
            std::cout << ',';
        }
        std::cout << '\n';
    }

    FAIL();
}

} // namespace util

#endif // NV_VPI_TEST_UTIL_GTEST_HPP
