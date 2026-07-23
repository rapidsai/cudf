/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use cudf_sys::{
    cudfTable_t, cudfTableDestroy, cudfTableGetColumn, cudfTableGetNumColumns, cudfTableGetNumRows,
};

use crate::column::Column;
use crate::error::{CudfError, check_cudf};

/// Safe owning wrapper around `cudfTable_t`.
pub struct Table {
    inner: cudfTable_t,
}

impl Table {
    /// Create an owning table from a raw handle.
    pub(crate) fn from_raw(inner: cudfTable_t) -> Self {
        Table { inner }
    }

    /// Return the raw cuDF table handle.
    pub fn as_raw(&self) -> cudfTable_t {
        self.inner
    }

    /// Return the number of columns in this table.
    pub fn num_columns(&self) -> Result<i32, CudfError> {
        let mut num_columns: i32 = 0;
        unsafe { check_cudf(cudfTableGetNumColumns(self.inner, &mut num_columns))? };
        Ok(num_columns)
    }

    /// Return the number of rows in this table.
    pub fn num_rows(&self) -> Result<i64, CudfError> {
        let mut num_rows: i64 = 0;
        unsafe { check_cudf(cudfTableGetNumRows(self.inner, &mut num_rows))? };
        Ok(num_rows)
    }

    /// Get an owning column extracted from this table.
    /// The caller owns the returned column independently and it will be
    /// destroyed via `cudfColumnDestroy` when dropped.
    pub fn column(&self, index: i32) -> Result<Column, CudfError> {
        let mut column = std::ptr::null_mut();
        unsafe { check_cudf(cudfTableGetColumn(self.inner, index, &mut column))? };
        Ok(Column::from_raw(column))
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                let _ = cudfTableDestroy(self.inner);
            }
        }
    }
}

unsafe impl Send for Table {}
