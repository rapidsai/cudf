/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Binary operations on GPU columns.

use cudf_sys::{cudfBinaryOp_t, cudfBinaryOpColumns, cudfDataType_t, cudfTypeId_t};

use crate::column::Column;
use crate::error::{CudfError, check_cudf};
use crate::resources::Resources;

/// Binary operation type.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum BinaryOp {
    Add = cudfBinaryOp_t::CUDF_BINARY_OP_ADD as u32,
    Sub = cudfBinaryOp_t::CUDF_BINARY_OP_SUB as u32,
    Mul = cudfBinaryOp_t::CUDF_BINARY_OP_MUL as u32,
    Div = cudfBinaryOp_t::CUDF_BINARY_OP_DIV as u32,
    TrueDiv = cudfBinaryOp_t::CUDF_BINARY_OP_TRUE_DIV as u32,
    FloorDiv = cudfBinaryOp_t::CUDF_BINARY_OP_FLOOR_DIV as u32,
    Mod = cudfBinaryOp_t::CUDF_BINARY_OP_MOD as u32,
    Equal = cudfBinaryOp_t::CUDF_BINARY_OP_EQUAL as u32,
    NotEqual = cudfBinaryOp_t::CUDF_BINARY_OP_NOT_EQUAL as u32,
    Less = cudfBinaryOp_t::CUDF_BINARY_OP_LESS as u32,
    Greater = cudfBinaryOp_t::CUDF_BINARY_OP_GREATER as u32,
    LessEqual = cudfBinaryOp_t::CUDF_BINARY_OP_LESS_EQUAL as u32,
    GreaterEqual = cudfBinaryOp_t::CUDF_BINARY_OP_GREATER_EQUAL as u32,
}

impl BinaryOp {
    fn to_ffi(self) -> cudfBinaryOp_t {
        match self {
            BinaryOp::Add => cudfBinaryOp_t::CUDF_BINARY_OP_ADD,
            BinaryOp::Sub => cudfBinaryOp_t::CUDF_BINARY_OP_SUB,
            BinaryOp::Mul => cudfBinaryOp_t::CUDF_BINARY_OP_MUL,
            BinaryOp::Div => cudfBinaryOp_t::CUDF_BINARY_OP_DIV,
            BinaryOp::TrueDiv => cudfBinaryOp_t::CUDF_BINARY_OP_TRUE_DIV,
            BinaryOp::FloorDiv => cudfBinaryOp_t::CUDF_BINARY_OP_FLOOR_DIV,
            BinaryOp::Mod => cudfBinaryOp_t::CUDF_BINARY_OP_MOD,
            BinaryOp::Equal => cudfBinaryOp_t::CUDF_BINARY_OP_EQUAL,
            BinaryOp::NotEqual => cudfBinaryOp_t::CUDF_BINARY_OP_NOT_EQUAL,
            BinaryOp::Less => cudfBinaryOp_t::CUDF_BINARY_OP_LESS,
            BinaryOp::Greater => cudfBinaryOp_t::CUDF_BINARY_OP_GREATER,
            BinaryOp::LessEqual => cudfBinaryOp_t::CUDF_BINARY_OP_LESS_EQUAL,
            BinaryOp::GreaterEqual => cudfBinaryOp_t::CUDF_BINARY_OP_GREATER_EQUAL,
        }
    }
}

/// cuDF data type descriptor.
#[derive(Debug, Copy, Clone)]
pub struct DataType {
    inner: cudfDataType_t,
}

impl DataType {
    /// Create a data type with no scale (non-decimal types).
    pub fn new(id: cudfTypeId_t) -> Self {
        DataType {
            inner: cudfDataType_t { id, scale: 0 },
        }
    }

    /// Create a decimal data type with a specified scale.
    pub fn with_scale(id: cudfTypeId_t, scale: i32) -> Self {
        DataType {
            inner: cudfDataType_t { id, scale },
        }
    }

    /// Return the raw FFI data type.
    pub fn as_raw(&self) -> cudfDataType_t {
        self.inner
    }
}

impl Column {
    /// Perform a binary operation between this column and `rhs`.
    pub fn binary_op(
        &self,
        rhs: &Column,
        op: BinaryOp,
        output_type: DataType,
        resources: &Resources,
    ) -> Result<Column, CudfError> {
        let mut result = std::ptr::null_mut();
        unsafe {
            check_cudf(cudfBinaryOpColumns(
                self.as_raw(),
                rhs.as_raw(),
                op.to_ffi(),
                output_type.as_raw(),
                resources.stream()?,
                &mut result,
            ))?;
        }
        Ok(Column::from_raw(result))
    }
}
