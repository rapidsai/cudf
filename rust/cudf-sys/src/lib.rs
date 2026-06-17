/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw FFI bindings to libcudf_c.

/// Opaque CUDA stream handle used by the cuDF C ABI.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _private: [u8; 0],
}

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut CUstream_st;

// Bindings are pre-generated and checked in at src/bindings.rs.
// Use `rust/scripts/generate-bindings.sh` to regenerate them.
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, unused_attributes)]
mod bindings;

pub use bindings::*;
