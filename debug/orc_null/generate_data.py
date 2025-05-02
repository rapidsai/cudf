import pandas as pd
import numpy as np


class DataGenerator:
    def __init__(self, orc_path):
        self.orc_path = orc_path
        self.df = None
        print("--> orc_path: {}".format(orc_path))

    def run(self):
        my_list = [np.nan] * 10000
        my_list.extend([-1] * 5)
        my_series = pd.Series(my_list, name="biu", dtype="Int64")
        df = pd.DataFrame(my_series)
        print(df)
        print(df["biu"].dtype)
        df.to_orc(self.orc_path)


if __name__ == "__main__":
    orc_path = "data/ref.orc"
    dg = DataGenerator(orc_path)
    dg.run()
