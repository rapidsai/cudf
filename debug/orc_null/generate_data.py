import pandas as pd
import numpy as np
import pyarrow


class DataGenerator:
    def __init__(self):
        pass

    def run_simple(self):
        orc_path = "data/simple.orc"
        my_list = [10] * 10000
        my_series_1 = pd.Series(my_list, dtype="Int64")

        df = pd.DataFrame({"col_a": my_series_1})
        print(df)
        print(df["col_a"].dtype)
        df.to_orc(orc_path, engine_kwargs={"stripe_size": 1024 * 8})

        tmp = pyarrow.orc.ORCFile(orc_path)
        print("Number of stripes: {:}".format(tmp.nstripes))

    def run_null(self):
        orc_path = "data/null.orc"

        # row group: all nulls
        my_list = [np.nan] * 10000

        # row group: alternate between non-null and null
        tmp = [10] * 10000
        for idx in range(len(tmp)):
            if idx % 2 == 1:
                tmp[idx] = np.nan
        my_list.extend(tmp)

        # row group: non-null
        my_list.extend([-1] * 5)

        my_series = pd.Series(my_list, dtype="Int64")

        df = pd.DataFrame({"col_a": my_series})
        print(df)
        print(df["col_a"].dtype)
        df.to_orc(orc_path)

    def run_comprehensive(self):
        orc_path = "data/comprehensive.orc"
        my_list = [10] * 30005
        my_series_1 = pd.Series(my_list, dtype="Int64")

        my_list = [10] * 10000
        my_list.extend([np.nan] * 10000)
        tmp = [10] * 10000
        for idx in range(len(tmp)):
            if idx % 2 == 1:
                tmp[idx] = np.nan
        my_list.extend(tmp)
        my_list.extend([10] * 5)
        my_series_2 = pd.Series(my_list, dtype="Int64")

        df = pd.DataFrame({"col_a": my_series_1, "col_b": my_series_2})
        print(df)
        print(df["col_a"].dtype)
        df.to_orc(orc_path, engine_kwargs={"stripe_size": 327680})

        tmp = pyarrow.orc.ORCFile(orc_path)
        print("Number of stripes: {:}".format(tmp.nstripes))


if __name__ == "__main__":
    dg = DataGenerator()
    dg.run_simple()
    dg.run_null()
    dg.run_comprehensive()
