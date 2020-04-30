import dask.order


def order(dsk, dependencies=None):
    """Key ordering optimized for cuDF"""

    ret = dask.order.order(dsk, dependencies)

    # TODO: Hack to priorities "shuffle-split"
    shuffle_split_keys = []
    for k in ret.keys():
        if len(k) > 0 and "shuffle-split" in k[0]:
            shuffle_split_keys.append(k)

    for k in shuffle_split_keys:
        ret[k] = 0

    return ret
