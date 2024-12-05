#!/usr/bin/env python3

# References:
# https://github.com/rapidsai/cudf/issues/17155
#
# In the shell, export KVIKIO_COMPAT_MODE=ON
#
# Col 'a' is the same for pandas and cudf.
# Col 'b' starts to differ at row 630107.

import pandas as pd
import cudf
import numpy as np
import datetime as dt


class CompareManager:
    def __init__(self, orcPath):
        self.orcPath = orcPath
        self.dfFull = None
        print("--> orcPath: {}".format(orcPath))

    # Reference: https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.timestamp.html#pandas.Timestamp.timestamp
    def getTime(self, timeStamp):
        if isinstance(timeStamp, pd.Timestamp):
            npTimeStamp = timeStamp.to_numpy()
        elif isinstance(timeStamp, np.datetime64):
            npTimeStamp = timeStamp

        tmp1 = npTimeStamp.astype("datetime64[ns]").astype("int")
        epochTimeElapsedSec = tmp1 // 1e9
        epochTimeElapsedNano = tmp1 / 1e9 - tmp1 // 1e9

        tmp2 = npTimeStamp - np.datetime64("2015-01-01")
        orcTimeElapsedSec = tmp2 // np.timedelta64(1, "s")
        orcTimeElapsedNano = tmp2 / np.timedelta64(1, "s") - orcTimeElapsedSec

        return (
            epochTimeElapsedSec,
            epochTimeElapsedNano,
            orcTimeElapsedSec,
            orcTimeElapsedNano,
        )

    def doIt(self):
        pdDf = pd.read_orc(self.orcPath)
        # cfDf = cudf.read_orc(self.orcPath, engine="pyarrow")
        cfDf = cudf.read_orc(self.orcPath, engine="cudf")

        print(pdDf.shape)
        print(cfDf.shape)

        # Convert to numpy array for speed
        b1 = pdDf["b"].to_numpy()
        b2 = cfDf["b"].to_numpy()

        targetIdx = -1
        for rowIdx in range(len(b1)):
            if b1[rowIdx] != b2[rowIdx]:
                print("Column b: {}".format(rowIdx))
                targetIdx = rowIdx
                break

        print("targetIdx: {}".format(targetIdx))

        if targetIdx == -1:
            print("Same data")
            return

        for k in range(-3, 10):
            # Now that the target index has been found,
            # use pandas/cudf's dataframe instead of numpy
            # for better timestamp handling
            ts1 = pdDf["b"][targetIdx + k]
            ts2 = cfDf["b"][targetIdx + k]
            if k == 0:
                print("-------------------------------------------------")

            res1 = self.getTime(ts1)
            res2 = self.getTime(ts2)
            print(
                "{}: {} ({}+{}) vs {} ({}+{})".format(
                    targetIdx + k, ts1, res1[2], res1[3], ts2, res2[2], res2[3]
                )
            )
            if k == 0:
                print("-------------------------------------------------")

        # lastRowGroupFirstRowIdx = 630000
        # if len(b1) <= lastRowGroupFirstRowIdx:
        #     print("--> Small data frame. diff.txt is not calcualted.")
        #     return

        # with open("diff.txt", "w") as f:
        #     secDiffCount = 0
        #     nanoDiffCount = 0
        #     for i in range(lastRowGroupFirstRowIdx, len(b1)):
        #         ts1 = pdDf["b"][i]
        #         ts2 = cfDf["b"][i]
        #         res1 = self.getTime(ts1)
        #         res2 = self.getTime(ts2)

        #         if abs(res1[2] - res2[2]) > 1:
        #             secDiffCount += 1

        #         if abs(res1[3] - res2[3]) > 1e-7:
        #             nanoDiffCount += 1

        #         f.write(
        #             "{}: {} ({}+{}) vs {} ({}+{})\n".format(
        #                 i, ts1, res1[2], res1[3], ts2, res2[2], res2[3]
        #             )
        #         )

        #     print("secDiffCount: {}".format(secDiffCount))
        #     print("nanoDiffCount: {}".format(nanoDiffCount))


if __name__ == "__main__":
    # orcPath = "data/timestamp_bug.snappy.orc"
    orcPath = "../col_b_only.orc"

    cmObj = CompareManager(orcPath)
    cmObj.doIt()
