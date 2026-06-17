/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use cudf_sys::{
    cudfColumn_t, cudfColumnDestroy, cudfColumnGetNullCount, cudfColumnGetSize, cudfColumnGetType,
    cudfDataType_t,
};

use crate::error::{CudfError, check_cudf};

/// Safe owning wrapper around `cudfColumn_t`.
pub struct Column {
    inner: cudfColumn_t,
    borrowed: bool,
}

impl Column {
    /// Create an owning column from a raw handle.
    pub(crate) fn from_raw(inner: cudfColumn_t) -> Self {
        Column {
            inner,
            borrowed: false,
        }
    }

    /// Create a non-owning column view from a raw handle.
    pub(crate) fn from_raw_borrowed(inner: cudfColumn_t) -> Self {
        Column {
            inner,
            borrowed: true,
        }
    }

    /// Return the raw cuDF column handle.
    pub fn as_raw(&self) -> cudfColumn_t {
        self.inner
    }

    /// Return the number of rows in this column.
    pub fn size(&self) -> Result<i64, CudfError> {
        let mut size: i64 = 0;
        unsafe { check_cudf(cudfColumnGetSize(self.inner, &mut size))? };
        Ok(size)
    }

    /// Return the null count for this column.
    pub fn null_count(&self) -> Result<i64, CudfError> {
        let mut null_count: i64 = 0;
        unsafe { check_cudf(cudfColumnGetNullCount(self.inner, &mut null_count))? };
        Ok(null_count)
    }

    /// Return the cuDF data type for this column.
    pub fn data_type(&self) -> Result<cudfDataType_t, CudfError> {
        let mut data_type = cudfDataType_t {
            id: cudf_sys::cudfTypeId_t::CUDF_TYPE_EMPTY,
            scale: 0,
        };
        unsafe { check_cudf(cudfColumnGetType(self.inner, &mut data_type))? };
        Ok(data_type)
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        if !self.borrowed && !self.inner.is_null() {
            unsafe {
                let _ = cudfColumnDestroy(self.inner);
            }
        }
    }
}

unsafe impl Send for Column {}
