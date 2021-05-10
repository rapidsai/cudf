# Copyright (c) 2021, NVIDIA CORPORATION.


class ListMethods:
    # TODO: Maybe use inheritance instead?
    def __init__(self, d_series):
        self.d_series = d_series

    def len(self):
        return self.d_series.map_partitions(
            lambda s: s.list.len(), meta=self.d_series._meta
        )

    def contains(self, search_key):
        return self.d_series.map_partitions(
            lambda s: s.list.contains(search_key), meta=self.d_series._meta
        )

    def get(self, index):
        return self.d_series.map_partitions(
            lambda s: s.list.get(index), meta=self.d_series._meta
        )

    @property
    def leaves(self):
        return self.d_series.map_partitions(
            lambda s: s.list.leaves, meta=self.d_series._meta
        )

    def take(self, lists_indices):
        return self.d_series.map_partitions(
            lambda s: s.list.take(lists_indices), meta=self.d_series._meta
        )

    def unique(self):
        return self.d_series.map_partitions(
            lambda s: s.list.unique(), meta=self.d_series._meta
        )

    def sort_values(
        self,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
    ):
        return self.d_series.map_partitions(
            lambda s: s.list.sort_values(
                ascending, inplace, kind, na_position, ignore_index
            ),
            meta=self.d_series._meta,
        )
