import sys

import cudf

import cudfkernel  # Cython bindings to execute existing CUDA Kernels


def read_df(weather_file_path):
    print(
        "Reading Weather file '"
        + weather_file_path
        + "' to Dataframe with Python"
    )

    # CSV reader options
    column_names = [
        "station_id",
        "date",
        "type",
        "val",
        "m_flag",
        "q_flag",
        "s_flag",
        "obs_time",
    ]
    usecols = column_names[0:4]

    # All 2010 weather recordings
    weather_df = cudf.read_csv(
        weather_file_path, names=column_names, usecols=usecols
    )

    # There are 5 possible recording types. PRCP, SNOW, SNWD, TMAX, TMIN
    # Rainfall is stored as 1/10ths of MM.
    rainfall_df = weather_df[weather_df["type"] == "PRCP"]
    print(rainfall_df.head())

    # Run the custom Kernel on the specified Dataframe Columns
    rainfall_kernel = cudfkernel.CudfWrapper(
        rainfall_df
    )  # Wrap the dataframe you want to perform Kernel calls on
    rainfall_kernel.tenth_mm_to_inches(
        4
    )  # column with index 4 is the "val" column

    # Shows head() after rainfall totals have been altered
    print(rainfall_df.head())


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Input Weather datafile path missing")

    read_df(sys.argv[1])
