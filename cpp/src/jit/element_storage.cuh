/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cudf {

// can store any nullable or non-nullable element type.
template <bool nullable, int max_element_size>
struct element_storage {
  alignas(max_element_size) unsigned char data[max_element_size * 2];
};

// can store any non-nullable element type.
//
template <int max_element_size>
struct element_storage<false, max_element_size> {
  alignas(max_element_size) unsigned char data[max_element_size];
};

}  // namespace cudf
