# This repository is for storing "synthetic" datasets for testing predicate
# pushdown.
#
# The purpose of synthetic datasets is for ensuring that when
# certain phenomena (e.g., cardinality) are exaggurated, we perform better
# than cuIO and pyarrow. We want to be able to note in what cases do we
# perform better.
#
# For general comparison, we should use "real" datasets. 

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prettytable import PrettyTable
from mimesis import Generic
g = Generic('es')

class ColumnParameters:
    # There might be a case where we want to try out different distributions
    # (i.e., something that isn't just uniform). For now, we just want to vary
    # cardinality and presence of nulls in different columns.
    #
    # ty is a callable function, specifically a Mimesis provider
    def __init__(self, cardinality=0.2, null_frequency=0.1, ty=g.food.fruit, is_sorted=False):
        self.cardinality = cardinality
        self.null_frequency = null_frequency
        self.ty = ty
        self.is_sorted = is_sorted

default_column_params = [ColumnParameters(ty=x) for x in [g.datetime.datetime, g.address.city, g.business.company_type, g.person.first_name]]

# The construction of these datasets should vary along the following
# dimensions-
# - Similarity between columns
# - Cardinality
# - Presence of null values
# - Type of data
# - Sorted
# - Size in # of rows
# - Size in # of columns
class Parameters:
    def __init__(self, num_rows=2048, num_cols=4, column_params=default_column_params):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.column_params = column_params

    def __str__(self):
        print("table with " + str(self.num_cols) + " columns and " + str(self.num_rows) + " rows:")
        params_info = PrettyTable()
        params_info.add_column("", ["type", "cardinality", "null frequency", "sorted?"])
        for i in range(self.num_cols):
            col_params = self.column_params[i]
            params_info.add_column("column " + str(i), [
                col_params.ty.__name__,
                '{:.1%}'.format(col_params.cardinality) + " unique",
                '{:.1%}'.format(col_params.null_frequency) + " null",
                "sorted" if col_params.is_sorted else "not sorted"
            ])
        return str(params_info)

# Synthesizes a new dataset and stores it in a Parquet file:
def synthesize(name, parameters):
    print("synthesizer with following parameters:")
    print(parameters)

    print("synthesizing...")
    data = {}
    columns_to_sort = []
    for i in range(parameters.num_cols):
        column_parameters = parameters.column_params[i]
        vals = np.array([column_parameters.ty() for _ in range(int(1 / column_parameters.cardinality))])
        generate_cell = lambda: None if np.random.rand() < column_parameters.null_frequency else np.random.choice(vals)
        data[str(i)] = np.array([generate_cell() for _ in range(parameters.num_rows)])
        if column_parameters.is_sorted:
             columns_to_sort.append(str(i))
    
    # Create DataFrame and sort columns appropriately
    df = pd.DataFrame(data)
    df = df.sort_values(columns_to_sort)

    # Store in Parquet file
    file_to_store_in = name + ".parquet" if not name.endswith(".parquet") else name
    tbl = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        tbl,
        root_path=file_to_store_in,
        row_group_size=64,
        partition_cols=['3'],
    )
    #df.to_parquet(file_to_store_in, allow_truncated_timestamps=True, row_group_size=512)
    #df.to_parquet(file_to_store_in, engine='fastparquet', file_scheme='hive')
    print("done")
    print("synthesized dataset stored in " + file_to_store_in)

default = Parameters()

# Focusing on the company type column...
very_sparse_column_params = list(default_column_params)
very_sparse_column_params[2].null_frequency = 0.8
very_sparse_column_params[2].cardinality = 1.0 / 16.0
very_sparse = Parameters(num_rows=2048 << 4, column_params=very_sparse_column_params)

# Focusing on the date column...
dates_sorted_column_params = list(default_column_params)
dates_sorted_column_params[0].is_sorted = True
dates_sorted_column_params[0].cardinality = 1.0 / (2048 << 4)
dates_sorted = Parameters(num_rows=2048, column_params=dates_sorted_column_params)
high_cardinality_column_params = list(default_column_params)
high_cardinality_column_params[0].is_sorted = True
high_cardinality_column_params[0].cardinality = 1.0 / (2048 << 5)
high_cardinality = Parameters(num_rows=2048, column_params=high_cardinality_column_params)

# Focusing on the city column...
low_cardinality_column_params = list(default_column_params)
low_cardinality_column_params[1].is_sorted = True
low_cardinality_column_params[1].cardinality = 1.0 / 20.0
low_cardinality = Parameters(num_rows=2048, column_params=low_cardinality_column_params)

# Simple
# This generates data for 2048 people regarding their age, city, industry,
# and name. There should be 4 row groups of 512 rows each. Sorting is by age.
simple_column_params = list(default_column_params)
simple_column_params[0].ty = g.person.age
simple_column_params[0].is_sorted = True
simple_column_params[1].is_sorted = False
simple = Parameters(num_rows=2048, column_params=simple_column_params)
