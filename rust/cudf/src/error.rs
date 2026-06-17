/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use cudf_sys::cudfError_t;

#[derive(Debug, thiserror::Error)]
pub enum CudfError {
    #[error("CUDF error: {0}")]
    Error(String),
    #[error("CUDF error (no details)")]
    Unknown,
}

/// Check a `cudfError_t` return value and return a Rust `Result`.
pub fn check_cudf(err: cudfError_t) -> Result<(), CudfError> {
    match err {
        cudfError_t::CUDF_SUCCESS => Ok(()),
        cudfError_t::CUDF_ERROR => {
            let msg = unsafe {
                let ptr = cudf_sys::cudfGetLastErrorText();
                if ptr.is_null() {
                    return Err(CudfError::Unknown);
                }
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            };
            Err(CudfError::Error(msg))
        }
    }
}
