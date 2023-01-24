# Copyright (c) 2020-2023, NVIDIA CORPORATION.
from typing import Any, Dict

import numba
import pandas as pd
from numba import cuda, types
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate, AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.np import numpy_support

index_default_type = numpy_support.from_dtype(
    pd.RangeIndex(0, 0).dtype
)  # int64
SUPPORTED_GROUPBY_NUMBA_TYPES = [types.int64, types.float64]
SUPPORTED_GROUPBY_NUMPY_TYPES = [
    numpy_support.as_dtype(dt) for dt in [types.int64, types.float64]
]


class Group:
    """
    A piece of python code whose purpose is to be replaced
    during compilation. After being registered to GroupType,
    serves as a handle for instantiating GroupType objects
    in python code and accessing their attributes
    """

    def __init__(self, group_data, size, index, dtype, index_dtype):
        self.group_data = group_data
        self.size = size
        self.index = index
        self.dtype = dtype
        self.index_dtype = index_dtype


class GroupType(numba.types.Type):
    """
    Numba extension type carrying metadata associated with a single
    GroupBy group. This metadata ultimately is passed to the CUDA
    __device__ function which actually performs the work.
    """

    def __init__(self, group_scalar_type, index_type=index_default_type):
        self.group_scalar_type = group_scalar_type
        self.index_type = index_type
        self.group_data_type = types.CPointer(group_scalar_type)
        self.size_type = types.int64
        self.group_index_type = types.CPointer(index_type)
        super().__init__(
            name=f"Group({self.group_scalar_type}, {self.index_type})"
        )


@typeof_impl.register(Group)
def typeof_group(val, c):
    """
    Tie Group and GroupType together such that when Numba
    sees usage of Group in raw python code, it knows to
    treat those usages as uses of GroupType
    """
    return GroupType(
        numba.np.numpy_support.from_dtype(val.dtype),
        numba.np.numpy_support.from_dtype(val.index_dtype),
    )


# The typing of the python "function" Group.__init__
# as it appears in python code
@type_callable(Group)
def type_group(context):
    def typer(group_data, size, index):
        if (
            isinstance(group_data, types.Array)
            and isinstance(size, types.Integer)
            and isinstance(index, types.Array)
        ):
            return GroupType(group_data.dtype, index.dtype)

    return typer


@register_model(GroupType)
class GroupModel(models.StructModel):
    def __init__(
        self, dmm, fe_type
    ):  # fe_type is fully instantiated group type
        members = [
            ("group_data", types.CPointer(fe_type.group_scalar_type)),
            ("size", types.int64),
            ("index", types.CPointer(fe_type.index_type)),
        ]
        super().__init__(dmm, fe_type, members)


call_cuda_functions: Dict[Any, Any] = {}


def _register_cuda_reduction_caller(func, inputty, retty):
    cuda_func = cuda.declare_device(
        f"Block{func}_{inputty}", retty(types.CPointer(inputty), types.int64)
    )

    def caller(data, size):
        return cuda_func(data, size)

    if call_cuda_functions.get(func.lower()) is None:
        call_cuda_functions[func.lower()] = {}

    type_key = (retty, inputty)
    call_cuda_functions[func.lower()][type_key] = caller


def _register_cuda_idxreduction_caller(func, inputty):
    cuda_func = cuda.declare_device(
        f"Block{func}_{inputty}",
        types.int64(
            types.CPointer(inputty), types.CPointer(types.int64), types.int64
        ),
    )

    def caller(data, index, size):
        return cuda_func(data, index, size)

    # only support default index type right now
    type_key = (index_default_type, inputty)
    if call_cuda_functions.get(func.lower()) is None:
        call_cuda_functions[func.lower()] = {}
    call_cuda_functions[func.lower()][type_key] = caller


def _create_reduction_attr(name, retty=None):
    class Attr(AbstractTemplate):
        key = name

    if retty:

        def generic(self, args, kws):
            return nb_signature(retty, recvr=self.this)

    else:

        def generic(self, args, kws):
            return nb_signature(self.this.group_scalar_type, recvr=self.this)

    Attr.generic = generic

    def _attr(self, mod):
        return types.BoundFunction(
            Attr, GroupType(mod.group_scalar_type, mod.index_type)
        )

    return _attr


class GroupIdxMax(AbstractTemplate):
    key = "GroupType.idxmax"

    def generic(self, args, kws):
        return nb_signature(self.this.index_type, recvr=self.this)


class GroupIdxMin(AbstractTemplate):
    key = "GroupType.idxmin"

    def generic(self, args, kws):
        return nb_signature(self.this.index_type, recvr=self.this)


@cuda_registry.register_attr
class GroupAttr(AttributeTemplate):
    key = GroupType

    resolve_max = _create_reduction_attr("GroupType.max")
    resolve_min = _create_reduction_attr("GroupType.min")
    resolve_sum = _create_reduction_attr("GroupType.sum")

    resolve_size = _create_reduction_attr("GroupType.size", retty=types.int64)
    resolve_count = _create_reduction_attr(
        "GroupType.count", retty=types.int64
    )
    resolve_mean = _create_reduction_attr(
        "GroupType.mean", retty=types.float64
    )
    resolve_var = _create_reduction_attr("GroupType.var", retty=types.float64)
    resolve_std = _create_reduction_attr("GroupType.std", retty=types.float64)

    def resolve_idxmax(self, mod):
        return types.BoundFunction(
            GroupIdxMax, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_idxmin(self, mod):
        return types.BoundFunction(
            GroupIdxMin, GroupType(mod.group_scalar_type, mod.index_type)
        )


_register_cuda_reduction_caller("Max", types.float64, types.float64)
_register_cuda_reduction_caller("Max", types.int64, types.int64)
_register_cuda_reduction_caller("Min", types.float64, types.float64)
_register_cuda_reduction_caller("Min", types.int64, types.int64)
_register_cuda_reduction_caller("Min", types.float64, types.float64)
_register_cuda_reduction_caller("Sum", types.int64, types.int64)
_register_cuda_reduction_caller("Sum", types.float64, types.float64)
_register_cuda_reduction_caller("Mean", types.int64, types.float64)
_register_cuda_reduction_caller("Mean", types.float64, types.float64)
_register_cuda_reduction_caller("Std", types.int64, types.float64)
_register_cuda_reduction_caller("Std", types.float64, types.float64)
_register_cuda_reduction_caller("Var", types.int64, types.float64)
_register_cuda_reduction_caller("Var", types.float64, types.float64)
_register_cuda_idxreduction_caller("IdxMax", types.int64)
_register_cuda_idxreduction_caller("IdxMax", types.float64)
_register_cuda_idxreduction_caller("IdxMin", types.int64)
_register_cuda_idxreduction_caller("IdxMin", types.float64)


make_attribute_wrapper(GroupType, "group_data", "group_data")
make_attribute_wrapper(GroupType, "index", "index")
make_attribute_wrapper(GroupType, "size", "size")
