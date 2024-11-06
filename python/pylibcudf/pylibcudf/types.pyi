# Copyright (c) 2024, NVIDIA CORPORATION.
from enum import IntEnum, auto

class Interpolation(IntEnum):
    LINEAR = auto()
    LOWER = auto()
    HIGHER = auto()
    MIDPOINT = auto()
    NEAREST = auto()

class MaskState(IntEnum):
    UNALLOCATED = auto()
    UNINITIALIZED = auto()
    ALL_VALID = auto()
    ALL_NULL = auto()

class NanEquality(IntEnum):
    ALL_EQUAL = auto()
    UNEQUAL = auto()

class NanPolicy(IntEnum):
    NAN_IS_NULL = auto()
    NAN_IS_VALID = auto()

class NullEquality(IntEnum):
    EQUAL = auto()
    UNEQUAL = auto()

class NullOrder(IntEnum):
    AFTER = auto()
    BEFORE = auto()

class NullPolicy(IntEnum):
    EXCLUDE = auto()
    INCLUDE = auto()

class Order(IntEnum):
    ASCENDING = auto()
    DESCENDING = auto()

class Sorted(IntEnum):
    NO = auto()
    YES = auto()

class TypeId(IntEnum):
    EMPTY = auto()
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    BOOL8 = auto()
    TIMESTAMP_DAYS = auto()
    TIMESTAMP_SECONDS = auto()
    TIMESTAMP_MILLISECONDS = auto()
    TIMESTAMP_MICROSECONDS = auto()
    TIMESTAMP_NANOSECONDS = auto()
    DURATION_DAYS = auto()
    DURATION_SECONDS = auto()
    DURATION_MILLISECONDS = auto()
    DURATION_MICROSECONDS = auto()
    DURATION_NANOSECONDS = auto()
    DICTIONARY32 = auto()
    STRING = auto()
    LIST = auto()
    DECIMAL32 = auto()
    DECIMAL64 = auto()
    DECIMAL128 = auto()
    STRUCT = auto()
    NUM_TYPE_IDS = auto()

class DataType:
    def __init__(self, type_id: TypeId, scale: int = 0) -> None: ...
    def id(self) -> TypeId: ...
    def scale(self) -> int: ...

def size_of(dtype: DataType) -> int: ...

SIZE_TYPE: DataType
SIZE_TYPE_ID: TypeId
