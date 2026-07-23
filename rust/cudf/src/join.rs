/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Join operations on GPU tables.

use cudf_sys::{cudfJoinType_t, cudfTableJoin};

use crate::error::{CudfError, check_cudf};
use crate::resources::Resources;
use crate::table::Table;

impl Table {
    /// Perform an inner join between this table and `right`.
    ///
    /// `left_on` and `right_on` specify column indices to join on and must
    /// have equal length.
    pub fn inner_join(
        &self,
        right: &Table,
        left_on: &[i32],
        right_on: &[i32],
        resources: &Resources,
    ) -> Result<Table, CudfError> {
        if left_on.len() != right_on.len() {
            return Err(CudfError::Error(
                "left_on and right_on must have equal length".to_string(),
            ));
        }
        let num_keys = left_on.len() as i32;
        let mut result = std::ptr::null_mut();
        unsafe {
            check_cudf(cudfTableJoin(
                self.as_raw(),
                right.as_raw(),
                left_on.as_ptr(),
                right_on.as_ptr(),
                num_keys,
                cudfJoinType_t::CUDF_JOIN_INNER,
                resources.stream()?,
                &mut result,
            ))?;
        }
        Ok(Table::from_raw(result))
    }
}
