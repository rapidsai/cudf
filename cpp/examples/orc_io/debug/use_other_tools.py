#!/usr/bin/env python3

# References:
# https://github.com/rapidsai/cudf/issues/17155

import pandas as pd


class UsePandas:
    def __init__(self, orcPath):
        self.orcPath = orcPath
        self.newOrcPath = "timestamp_ok.snappy.orc"
        self.dfFull = None

    def read(self):
        self.dfFull = pd.read_orc(self.orcPath)

        # df = self.dfFull[self.dfFull.a == 10]
        # print(df)
        # print(df.shape)

        print(self.dfFull)
        print(self.dfFull.shape)

    def write(self):
        self.dfFull.to_orc(self.newOrcPath)


if __name__ == '__main__':
    orcPath = "../timestamp_bug.snappy.orc"

    pdObj = UsePandas(orcPath)
    pdObj.read()
    pdObj.write()
