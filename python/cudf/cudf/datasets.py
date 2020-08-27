import random
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker

import cudf

__all__ = ["timeseries", "randomdata"]


# TODO:
# change default of name from category to str type when nvstring are merged
def timeseries(
    start="2000-01-01",
    end="2000-01-31",
    freq="1s",
    dtypes={"name": "category", "id": int, "x": float, "y": float},
    seed=None,
):
    """ Create timeseries dataframe with random data

    Parameters
    ----------
    start : datetime (or datetime-like string)
        Start of time series
    end : datetime (or datetime-like string)
        End of time series
    dtypes : dict
        Mapping of column names to types.
        Valid types include {float, int, str, 'category'}
    freq : string
        String like '2s' or '1H' or '12W' for the time series frequency
    seed : int (optional)
        Randomstate seed

    Examples
    --------
    >>> import cudf as gd
    >>> gdf = gd.datasets.timeseries()
    >>> gdf.head()  # doctest: +SKIP
              timestamp    id     name         x         y
    2000-01-01 00:00:00   967    Jerry -0.031348 -0.040633
    2000-01-01 00:00:01  1066  Michael -0.262136  0.307107
    2000-01-01 00:00:02   988    Wendy -0.526331  0.128641
    2000-01-01 00:00:03  1016   Yvonne  0.620456  0.767270
    2000-01-01 00:00:04   998   Ursula  0.684902 -0.463278
    """

    index = pd.DatetimeIndex(
        pd.date_range(start, end, freq=freq, name="timestamp")
    )
    state = np.random.RandomState(seed)
    columns = dict(
        (k, make[dt](len(index), state)) for k, dt in dtypes.items()
    )
    df = pd.DataFrame(columns, index=index, columns=sorted(columns))
    if df.index[-1] == end:
        df = df.iloc[:-1]
    return cudf.from_pandas(df)


def randomdata(
    nrows=10, dtypes={"id": int, "x": float, "y": float}, seed=None
):
    """ Create a dataframe with random data

    Parameters
    ----------
    nrows : int
        number of rows in the dataframe
    dtypes : dict
        Mapping of column names to types.
        Valid types include {float, int, str, 'category'}
    seed : int (optional)
        Randomstate seed

    Examples
    --------
    >>> import cudf as gd
    >>> gdf = gd.datasets.randomdata()
    >>> cdf.head()  # doctest: +SKIP
            id                  x                   y
    0  1014 0.28361267466770146 -0.44274170661264334
    1  1026 -0.9937981936047235 -0.09433464773262323
    2  1038 -0.1266722796765325 0.20971126368240123
    3  1002 0.9280495300010041  0.5137701393017848
    4   976 0.9089527839187654  0.9881063385586304
    """
    state = np.random.RandomState(seed)
    columns = dict((k, make[dt](nrows, state)) for k, dt in dtypes.items())
    df = pd.DataFrame(columns, columns=sorted(columns))
    return cudf.from_pandas(df)


def make_float(n, rstate):
    return rstate.rand(n) * 2 - 1


def make_int(n, rstate):
    return rstate.poisson(1000, size=n)


names = [
    "Alice",
    "Bob",
    "Charlie",
    "Dan",
    "Edith",
    "Frank",
    "George",
    "Hannah",
    "Ingrid",
    "Jerry",
    "Kevin",
    "Laura",
    "Michael",
    "Norbert",
    "Oliver",
    "Patricia",
    "Quinn",
    "Ray",
    "Sarah",
    "Tim",
    "Ursula",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zelda",
]


def make_string(n, rstate):
    return rstate.choice(names, size=n)


def make_categorical(n, rstate):
    return pd.Categorical.from_codes(
        rstate.randint(0, len(names), size=n), names
    )


def make_bool(n, rstate):
    return rstate.choice([True, False], size=n)


make = {
    float: make_float,
    int: make_int,
    str: make_string,
    object: make_string,
    "category": make_categorical,
    bool: make_bool,
}


def get_rand_int(
    rows, seed=0, dtype=np.dtype("int64"), name=None, null_frequency=0.2
):
    dtype_info = np.iinfo(dtype)
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)
    int_values = [
        fake.pyint(min_value=dtype_info.min, max_value=dtype_info.max)
        for _ in range(rows)
    ]
    random_index = np.arange(rows)
    np.random.shuffle(random_index)
    random_index = random_index[: int(null_frequency * rows)]

    series = cudf.Series(int_values, name=name)
    series.iloc[random_index] = None
    return series


def get_rand_float(
    rows, seed=0, dtype=np.dtype("float64"), name=None, null_frequency=0.2
):
    dtype_info = np.finfo(dtype)
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)
    int_values = [
        fake.pyfloat(min_value=dtype_info.min, max_value=dtype_info.max)
        for _ in range(rows)
    ]
    random_index = np.arange(rows)
    np.random.shuffle(random_index)
    random_index = random_index[: int(null_frequency * rows)]

    series = cudf.Series(int_values, name=name)
    series.iloc[random_index] = None
    return series


def get_rand_bool(
    rows, seed=0, dtype=np.dtype("bool"), name=None, null_frequency=0.2
):
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)
    bool_values = [
        fake.boolean(chance_of_getting_true=50) for _ in range(rows)
    ]
    random_index = np.arange(rows)
    np.random.shuffle(random_index)
    random_index = random_index[: int(null_frequency * rows)]

    series = cudf.Series(bool_values, name=name)
    series.iloc[random_index] = None
    return series


def get_rand_str(
    rows,
    seed=0,
    dtype=np.dtype("object"),
    name=None,
    null_frequency=0.2,
    min_str_length=0,
    max_str_length=20,
    get_text=False,
):
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)

    if get_text:
        str_values = fake.texts(nb_texts=rows)
    else:
        str_values = [
            fake.pystr(min_chars=min_str_length, max_chars=max_str_length)
            for _ in range(rows)
        ]
    random_index = np.arange(rows)
    np.random.shuffle(random_index)
    random_index = random_index[: int(null_frequency * rows)]

    series = cudf.Series(str_values, dtype=dtype, name=name)
    series.iloc[random_index] = None
    return series


def get_rand_datetime(
    rows,
    seed=0,
    dtype=np.dtype("datetime64[ns]"),
    name=None,
    null_frequency=0.2,
    start_date=datetime(year=1970, month=1, day=1),
    end_date=pd.Timestamp.max.to_pydatetime(),
):
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)
    date_values = [
        fake.date_time_ad(start_datetime=start_date, end_datetime=end_date)
        for _ in range(rows)
    ]
    random_index = np.arange(rows)
    np.random.shuffle(random_index)
    random_index = random_index[: int(null_frequency * rows)]

    series = cudf.Series(date_values, dtype=dtype, name=name)
    series.iloc[random_index] = None
    return series


def get_rand_timedelta(
    rows,
    seed=0,
    dtype=np.dtype("timedelta64[ns]"),
    name=None,
    null_frequency=0.2,
    min_int=np.iinfo("int64").min + 1,
    max_int=np.iinfo("int64").max,
):
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)
    int_values = [
        fake.pyint(min_value=min_int, max_value=max_int) for _ in range(rows)
    ]
    random_index = np.arange(rows)
    np.random.shuffle(random_index)
    random_index = random_index[: int(null_frequency * rows)]

    series = cudf.Series(int_values, dtype=dtype, name=name)
    series.iloc[random_index] = None
    return series


def get_rand_category(
    rows, seed=0, name=None, null_frequency=0.2, cardinality=970
):
    if cardinality > 970:
        raise ValueError(
            "Maximum allowed words to choose from is 970."
            " Use get_rand_str with min and max words instead."
        )

    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    null_count = int(null_frequency * rows)
    values_count = rows - null_count

    category_values = (
        fake.words(nb=values_count, unique=False) + [None] * null_count
    )
    random.shuffle(category_values)

    series = cudf.Series(category_values, dtype="category", name=name)

    return series
