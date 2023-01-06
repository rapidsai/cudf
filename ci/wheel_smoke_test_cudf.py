# Copyright (c) 2022-2023, NVIDIA CORPORATION.

import cudf
import pyarrow as pa

if __name__ == '__main__':
    n_legs = pa.array([2, 4, 5, 100])
    animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
    names = ["n_legs", "animals"]
    foo = pa.table([n_legs, animals], names=names)
    df = cudf.DataFrame.from_arrow(foo)
    assert df.loc[df["animals"] == "Centipede"]["n_legs"].iloc[0] == 100
    assert df.loc[df["animals"] == "Flamingo"]["n_legs"].iloc[0] == 2
