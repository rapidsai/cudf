# nvstrings Timestamp Conversion Features

nvstrings currently supports converting timestamps to and from integer values
in specific units. The timestamp format is parsed using specifiers based on the
[strftime and strptime documentation](https://docs.python.org/3.7/library/datetime.html#strftime-and-strptime-behavior).

This document details supported specifiers and behaviour in `nvstrings.int2timestamp` and `nvstrings.timestamp2int` methods.
The specifiers are required for both parsing and formatting strings during conversion.

The following specifiers are supported as described below.

| Specifier | Description |
| :-------: | ----------- |
| %d | Day of the month: 01-31 |
| %m | Month of the year: 01-12 |
| %y | Year without century: 00-99 |
| %Y | Year with century: 0001-9999 |
| %H | 24-hour of the day: 00-23 |
| %I | 12-hour of the day: 01-12 |
| %M | Minute of the hour: 00-59|
| %S | Second of the minute: 00-59 |
| %f | 6-digit microsecond: 000000-999999 |
| %z | UTC offset with format ±HHMM[SS[.ffffff]]. Example +0500 |
| %j | Day of the year: 001-366 |
| %p | Only 'AM', 'PM' or 'am', 'pm' are recognized |

Other specifiers are not currently supported.

The nvstrings `int2timestamp` and `timestamp2int` method also allow
the following time units to be specified as a separate parameter:

| Specifier | Unit |
| :-------: | ---- |
| Y | years |
| M | months |
| D | days |
| h | hours |
| m | minutes |
| s | seconds |
| ms | milliseconds |
| us | microseconds |
| ns | nanoseconds |

These units are used in interpreting the integer values during conversion.
Regardless of units, these values must by of type uint64 (64-bit unsigned integer).
