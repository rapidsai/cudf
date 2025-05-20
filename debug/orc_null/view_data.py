import pandas as pd
import cudf


class ViewManager:
    def __init__(self, orc_path, csv_path):
        self.orc_path = orc_path
        self.csv_path = csv_path
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
            print(">>> pandas exception!!!")

        try:
            print("--> cudf")
            df = cudf.read_orc(self.orc_path)
            print(df)
            print(df["_col0"].dtype)
            df.to_csv(self.csv_path)
        except Exception as inst:
            print(inst)
            print(">>> cudf exception!!!")


if __name__ == "__main__":
    good_orc_path = "data/good_OrcEmptyRowGroup.orc"
    good_csv_path = "good.csv"
    good_obj = ViewManager(good_orc_path, good_csv_path)
    good_obj.run()

    special_orc_path = "data/special_OrcEmptyRowGroup.orc"
    special_csv_path = "special.csv"
    special_obj = ViewManager(special_orc_path, special_csv_path)
    special_obj.run()

    bad_orc_path = "data/bad_OrcEmptyRowGroup.orc"
    bad_csv_path = "bad.csv"
    bad_obj = ViewManager(bad_orc_path, bad_csv_path)
    bad_obj.run()

    bad_orc_path = "data/bad_2_OrcEmptyRowGroup.orc"
    bad_csv_path = "bad_2.csv"
    bad_obj = ViewManager(bad_orc_path, bad_csv_path)
    bad_obj.run()

    bad_orc_path = "data/bad_3_OrcEmptyRowGroup.orc"
    bad_csv_path = "bad_3.csv"
    bad_obj = ViewManager(bad_orc_path, bad_csv_path)
    bad_obj.run()

    bad_orc_path = "data/bad_4_OrcEmptyRowGroup.orc"
    bad_csv_path = "bad_4.csv"
    bad_obj = ViewManager(bad_orc_path, bad_csv_path)
    bad_obj.run()
