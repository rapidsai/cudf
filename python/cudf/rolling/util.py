from cudf.rolling.rolling import Rolling


def diff(in_arr, n):
    if n < 0:
        return Rolling(1, in_arr, forward_window=-n).forward_diff()
    elif n > 0:
        return Rolling(n + 1, in_arr).backward_diff()
    else:
        return in_arr


def shift(in_arr, n):
    if n < 0:
        return Rolling(1, in_arr, forward_window=-n).forward_shift()
    elif n > 0:
        return Rolling(n + 1, in_arr).backward_shift()
    else:
        return in_arr
