/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use cudf_sys::{
    cudaStream_t, cudfResources_t, cudfResourcesCreate, cudfResourcesDestroy, cudfStreamGet,
};

use crate::error::{CudfError, check_cudf};

/// RAII wrapper around `cudfResources_t`.
pub struct Resources {
    inner: cudfResources_t,
}

impl Resources {
    /// Create a new cuDF resources handle.
    pub fn new() -> Result<Self, CudfError> {
        let mut res: cudfResources_t = 0;
        unsafe { check_cudf(cudfResourcesCreate(&mut res))? };
        Ok(Resources { inner: res })
    }

    /// Return the raw cuDF resources handle.
    pub fn as_raw(&self) -> cudfResources_t {
        self.inner
    }

    /// Set the CUDA stream used by this resources handle.
    pub fn set_stream(&self, stream: cudaStream_t) -> Result<(), CudfError> {
        unsafe { check_cudf(cudf_sys::cudfStreamSet(self.inner, stream)) }
    }

    /// Get the CUDA stream associated with this resources handle.
    pub fn stream(&self) -> Result<cudaStream_t, CudfError> {
        let mut s: cudaStream_t = std::ptr::null_mut();
        unsafe { check_cudf(cudfStreamGet(self.inner, &mut s))? };
        Ok(s)
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        unsafe {
            let _ = cudfResourcesDestroy(self.inner);
        }
    }
}

// Resources owns a GPU context handle that can be transferred across threads.
unsafe impl Send for Resources {}
