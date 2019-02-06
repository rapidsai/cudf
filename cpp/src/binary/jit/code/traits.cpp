/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

namespace gdf {
namespace binops {
namespace jit {
namespace code {

const char* traits =
R"***(
#pragma once

    struct IntegralSigned {};

    struct IntegralUnsigned {};


    template <typename Type>
    constexpr bool isIntegral = false;

    template <>
    constexpr bool isIntegral<int8_t> = true;

    template <>
    constexpr bool isIntegral<int16_t> = true;

    template <>
    constexpr bool isIntegral<int32_t> = true;

    template <>
    constexpr bool isIntegral<int64_t> = true;

    template <>
    constexpr bool isIntegral<uint8_t> = true;

    template <>
    constexpr bool isIntegral<uint16_t> = true;

    template <>
    constexpr bool isIntegral<uint32_t> = true;

    template <>
    constexpr bool isIntegral<uint64_t> = true;



    template <typename Type>
    constexpr bool isFloatingPoint = false;

    template <>
    constexpr bool isFloatingPoint<float> = true;

    template <>
    constexpr bool isFloatingPoint<double> = true;



    template <typename Type>
    constexpr bool isIntegralSigned = false;

    template <>
    constexpr bool isIntegralSigned<int8_t> = true;

    template <>
    constexpr bool isIntegralSigned<int16_t> = true;

    template <>
    constexpr bool isIntegralSigned<int32_t> = true;

    template <>
    constexpr bool isIntegralSigned<int64_t> = true;



    template <typename Type>
    constexpr bool isIntegralUnsigned = false;

    template <>
    constexpr bool isIntegralUnsigned<uint8_t> = true;

    template <>
    constexpr bool isIntegralUnsigned<uint16_t> = true;

    template <>
    constexpr bool isIntegralUnsigned<uint32_t> = true;

    template <>
    constexpr bool isIntegralUnsigned<uint64_t> = true;


    template <typename Type>
    constexpr bool isFloat = false;

    template <>
    constexpr bool isFloat<float> = true;


    template <typename Type>
    constexpr bool isDouble = false;

    template <>
    constexpr bool isDouble<double> = true;


    template <typename X, typename Y>
    constexpr int MaxSize = ((sizeof(X) < sizeof(Y)) ? sizeof(Y) : sizeof(X));

    template <int N, typename T>
    struct HelperIntegralMap;

    template <>
    struct HelperIntegralMap<1, IntegralSigned> {
        using Type = int8_t;
    };

    template <>
    struct HelperIntegralMap<2, IntegralSigned> {
        using Type = int16_t;
    };

    template <>
    struct HelperIntegralMap<4, IntegralSigned> {
        using Type = int32_t;
    };

    template <>
    struct HelperIntegralMap<8, IntegralSigned> {
        using Type = int64_t;
    };

    template <>
    struct HelperIntegralMap<1, IntegralUnsigned> {
        using Type = uint8_t;
    };

    template <>
    struct HelperIntegralMap<2, IntegralUnsigned> {
        using Type = uint16_t;
    };

    template <>
    struct HelperIntegralMap<4, IntegralUnsigned> {
        using Type = uint32_t;
    };

    template <>
    struct HelperIntegralMap<8, IntegralUnsigned> {
        using Type = uint64_t;
    };

    template <int N, typename T>
    using IntegralMap = typename HelperIntegralMap<N, T>::Type;



    template <bool, typename, typename>
    struct helperIf;

    template <typename T, typename F>
    struct helperIf<true, T, F> {
        using type = T;
    };

    template <typename T, typename F>
    struct helperIf<false, T, F> {
        using type = F;
    };

    template <bool B, typename T, typename F>
    using If = typename helperIf<B, T, F>::type;



    template<bool B, class T>
    struct helperEnableIf
    {};

    template<class T>
    struct helperEnableIf<true, T> {
        using type = T;
    };

    template <bool B, class T = void>
    using enableIf = typename helperEnableIf<B,T>::type;



    template <typename Vax, typename Vay, typename = void>
    struct HelperCommonNumber {};

    template <typename Vax, typename Vay>
    struct HelperCommonNumber<Vax, Vay, enableIf<(isFloatingPoint<Vax> || isFloatingPoint<Vay>)>> {
        using Type = If<(sizeof(Vax) == 8 || sizeof(Vay) == 8), double, float>;
    };

    template <typename Vax, typename Vay>
    struct HelperCommonNumber<Vax, Vay, enableIf<(isIntegralSigned<Vax> && isIntegralSigned<Vay>)>> {
        using Type = IntegralMap<(MaxSize<Vax, Vay>), IntegralSigned>;
    };

    template <typename Vax, typename Vay>
    struct HelperCommonNumber<Vax, Vay, enableIf<(isIntegralSigned<Vax> && isIntegralUnsigned<Vay>)>> {
        using Type = IntegralMap<(MaxSize<Vax, Vay>), IntegralSigned>;
    };

    template <typename Vax, typename Vay>
    struct HelperCommonNumber<Vax, Vay, enableIf<(isIntegralUnsigned<Vax> && isIntegralSigned<Vay>)>> {
        using Type = IntegralMap<(MaxSize<Vax, Vay>), IntegralSigned>;
    };

    template <typename Vax, typename Vay>
    struct HelperCommonNumber<Vax, Vay, enableIf<(isIntegralUnsigned<Vax> && isIntegralUnsigned<Vay>)>> {
        using Type = IntegralMap<(MaxSize<Vax, Vay>), IntegralUnsigned>;
    };

    template <typename Vax, typename Vay>
    using CommonNumber = typename HelperCommonNumber<Vax, Vay>::Type;
)***";

} // namespace code
} // namespace jit
} // namespace binops
} // namespace gdf
