import pandas as pd
import cudf


class ViewManager:
    def __init__(self, orc_path):
        self.orc_path = orc_path
        self.df = None
        print("--> orc_path: {}".format(orc_path))

    def run(self):
        try:
            print("--> pandas")
            df = pd.read_orc(self.orc_path)
            print(df)
            print(df["_col0"].dtype)
        except Exception as inst:
            print(inst)

        try:
            print("--> cudf")
            df = cudf.read_orc(self.orc_path)
            print(df)
            print(df["_col0"].dtype)
        except Exception as inst:
            print(inst)


if __name__ == "__main__":
    orc_path = "data/good_OrcEmptyRowGroup.orc"
    good_obj = ViewManager(orc_path)
    good_obj.run()

    orc_path = "data/bad_OrcEmptyRowGroup.orc"
    bad_obj = ViewManager(orc_path)
    bad_obj.run()
