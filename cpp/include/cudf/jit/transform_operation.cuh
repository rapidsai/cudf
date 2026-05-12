/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

/// @brief The compute operation to perform on each element. This is a generic entry point for n-ary
/// transform operations.
/// @param user_data Pointer to user data passed to the kernel
/// @param element_index The index of the element to compute
/// @param inputs Pointer to the input elements for this operation; the caller guarantees the memory
/// layout and type of these elements based on the input column device views and input strides
/// @param input_stride The stride (in bytes) between consecutive input elements for a given input
/// column
/// @param outputs Pointer to the output elements for this operation; the caller guarantees the
/// memory layout and type of these elements based on the output column device views
/// @param output_stride The stride (in bytes) between consecutive output elements for a given
/// output column
/// @return An integer status code
extern "C" __device__ int cudf_transform_operation(void* __restrict__ user_data,
                                                   long int element_index,
                                                   void const* __restrict__ inputs,
                                                   int input_stride,
                                                   void* __restrict__ outputs,
                                                   int output_stride);
