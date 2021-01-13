import more_itertools
import pyarrow as pa

from cudf import _lib as libcudf
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase
from cudf.core.dtypes import DecimalDtype
from cudf.utils.utils import pa_mask_buffer_to_mask


class DecimalColumn(ColumnBase):
    @classmethod
    def from_arrow(cls, data: pa.Array):
        mask_buf = data.buffers()[0]
        mask = (
            pa_mask_buffer_to_mask(mask_buf, len(data))
            if mask_buf is not None
            else None
        )
        bts = data.buffers()[1].to_pybytes()
        bts = b"".join(list(more_itertools.sliced(bts, 8))[::2])
        return cls(
            data=Buffer.from_bytes(bts),
            size=len(data),
            dtype=DecimalDtype.from_arrow(data.type),
            mask=mask,
        )

    def to_arrow(self):
        data_buf_64 = self.base_data.to_host_array()
        zeros_buf = bytes(data_buf_64.size)
        data_buf_128 = pa.py_buffer(
            bytes(
                more_itertools.flatten(
                    more_itertools.interleave(
                        more_itertools.chunked(data_buf_64, 8),
                        more_itertools.chunked(zeros_buf, 8),
                    )
                )
            )
        )
        mask_buf = (
            pa.py_buffer(bytes(self.base_mask.to_host_array()))
            if self.base_mask is not None
            else None
        )
        return pa.Array.from_buffers(
            type=self.dtype.to_arrow(),
            length=self.size,
            buffers=[mask_buf, data_buf_128],
        )

    def binary_operator(self, op, other, reflect=False):
        if reflect:
            self, other = other, self
        result = libcudf.binaryop.binaryop(self, other, op, "int32")
        result.dtype.precision = binop_precision(self.dtype, other.dtype, op)
        return result


def binop_precision(l_dtype, r_dtype, op):
    p1, p2 = l_dtype.precision, r_dtype.precision
    s1, s2 = l_dtype.scale, r_dtype.scale
    if op in ("add", "sub"):
        return max(s1, s2) + max(p1 - s1, p2 - s2) + 1
    elif op == "mul":
        return p1 + p2 + 1
    else:
        raise NotImplementedError()
    # TODO extend with -, *, / + unit tests
    # TODO think about TRUE_DIV vs FLOOR_DIV vs (DIV only in C++))
    # TODO DIVISION rule -> p1 - s1 + s2 + max(6, s1 + p2 + 1)
    #     just do what C++ does -> highlight in review so Keith can comment
