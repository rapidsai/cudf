/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NV_VPI_PRIV_COMMON_COMPILER_HPP
#define NV_VPI_PRIV_COMMON_COMPILER_HPP

#if defined(__GNUC__) && !defined(__clang__)
#   define VPI_GCC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
#endif

#if defined(__clang__)
#   define VPI_CLANG_VERSION (__clang_major__*10000 + __clang_minor__*100 + __clang_patchlevel__)
#endif

#endif // NV_VPI_PRIV_COMMON_COMPILER_HPP


