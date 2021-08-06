from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

cimport cudf._lib.cpp.datetime as libcudf_datetime
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


def add_months(Column col, Column months):
    # months must be int16 dtype
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()
    cdef column_view months_view = months.view()

    with nogil:
        c_result = move(
            libcudf_datetime.add_calendrical_months(
                col_view,
                months_view
            )
        )

    return Column.from_unique_ptr(move(c_result))


def extract_datetime_component(Column col, object field):

    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()

    with nogil:
        if field == "year":
            c_result = move(libcudf_datetime.extract_year(col_view))
        elif field == "month":
            c_result = move(libcudf_datetime.extract_month(col_view))
        elif field == "day":
            c_result = move(libcudf_datetime.extract_day(col_view))
        elif field == "weekday":
            c_result = move(libcudf_datetime.extract_weekday(col_view))
        elif field == "hour":
            c_result = move(libcudf_datetime.extract_hour(col_view))
        elif field == "minute":
            c_result = move(libcudf_datetime.extract_minute(col_view))
        elif field == "second":
            c_result = move(libcudf_datetime.extract_second(col_view))
        elif field == "day_of_year":
            c_result = move(libcudf_datetime.day_of_year(col_view))
        else:
            raise ValueError(f"Invalid datetime field: '{field}'")

    result = Column.from_unique_ptr(move(c_result))

    if field == "weekday":
        # Pandas counts Monday-Sunday as 0-6
        # while we count Monday-Sunday as 1-7
        result = result.binary_operator("sub", result.dtype.type(1))

    return result


def is_leap_year(Column col):
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()

    with nogil:
        c_result = move(libcudf_datetime.is_leap_year(col_view))

    return Column.from_unique_ptr(move(c_result))


def last_day_of_month(Column col):
    cdef unique_ptr[column] c_result
    cdef column_view col_view = col.view()

    with nogil:
        c_result = move(libcudf_datetime.last_day_of_month(col_view))

    return Column.from_unique_ptr(move(c_result))
