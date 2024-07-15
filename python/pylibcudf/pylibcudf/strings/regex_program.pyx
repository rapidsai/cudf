# Copyright (c) 2024, NVIDIA CORPORATION.


from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from pylibcudf.libcudf.strings.regex_program cimport regex_program

from pylibcudf.strings.regex_flags import RegexFlags

from pylibcudf.strings.regex_flags cimport regex_flags


cdef class RegexProgram:

    def __init__(self, *args, **kwargs):
        raise ValueError("Do not instantiate RegexProgram directly, use create")

    @staticmethod
    def create(str pattern, int flags):
        cdef unique_ptr[regex_program] c_prog
        cdef regex_flags c_flags
        cdef string c_pattern = pattern.encode()

        cdef RegexProgram ret = RegexProgram.__new__(RegexProgram)
        if isinstance(flags, object):
            if isinstance(flags, (int, RegexFlags)):
                c_flags = <regex_flags>flags
                with nogil:
                    c_prog = regex_program.create(c_pattern, c_flags)

                ret.c_obj = move(c_prog)
            else:
                raise ValueError("flags must be of type RegexFlags")

        return ret
