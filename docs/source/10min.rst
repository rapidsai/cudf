10 Minutes to cuDF
=======================

Modeled after 10 Minutes to Pandas, this is a short introduction to cuDF, geared mainly for new users.

.. ipython:: python
   :suppress:

   import os
   import numpy as np
   import pandas as pd
   import cudf
   np.random.seed(12)

   #### Portions of this were borrowed from the
   #### cuDF cheatsheet and existing documentation.
   #### Created November, 2018.


Object Creation
---------------

Creating a `Series`.

.. ipython:: python

    s = cudf.Series([1,2,3,None,4])
    print(s)

Creating a `DataFrame` by specifying values for each column

.. ipython:: python

    df = cudf.DataFrame([('a', list(range(20))),
    ('b', list(reversed(range(20)))),
    ('c', list(range(20)))])
    print(df)


Creating a `Dataframe` from a pandas Dataframe. 

.. ipython:: python

    pdf = pd.DataFrame({'a': [0, 1, 2, 3],'b': [0.1, 0.2, None, 0.3]})
    gdf = cudf.DataFrame.from_pandas(pdf)
    print(gdf)


Viewing Data
-------------

Viewing the top rows of the GPU dataframe.

.. ipython:: python

    print(df.head(2))

Sorting by values.

.. ipython:: python

    print(df.sort_values(by='a', ascending=False))


Selection
------------

Getting
~~~~~~~~~~~~~~

Selecting a single column, which yields a `cudf.Series`, equivalent to `df.a`.

.. ipython:: python

    print(df['a'])



Selection by Label
~~~~~~~~~~~~~~~~~~~~~

Selecting rows from index 2 to index 5 from columns 'a' and 'b'.

.. ipython:: python

    print(df.loc[2:5, ['a', 'b']])



Selection by Position
~~~~~~~~~~~~~~~~~~~~~~

Selecting by integer slicing, like numpy/pandas.

.. ipython:: python

    print(df[3:5])

Selecting elements of a `Series` with direct index access.

.. ipython:: python

    print(s[2])


Boolean Indexing
~~~~~~~~~~~~~~~~~~~~~

Selecting rows in a `Series` by direct boolean indexing, if there are no missing values.

.. ipython:: python

    print(df.b[df.b > 15])


Selecting values from a DataFrame where a boolean condition is met, via the `query` API.

.. ipython:: python

    print(df.query("b == 3"))

Supported logical operators include `>`, `<`, `>=`, `<=`, `==`, and `!=`.


Setting
~~~~~~~~~~


Missing Data
------------

Missing data can be replaced by using the `fillna` method.

.. ipython:: python

    print(s.fillna(999))


Operations
------------

Stats
~~~~~~~~~

Calculating descriptive statistics (operations in general exclude missing data).

.. ipython:: python

    print(s.mean(), s.var(), s.sum_of_squares())


Applymap
~~~~~~~~~

Applying functions to a `Series`.

.. ipython:: python

    def add_ten(num):
        return num + 10

    print(df['a'].applymap(add_ten))


Histogramming
~~~~~~~~~~~~~~~~~~~~~

Counting the number of rows with each unique value of variable

.. ipython:: python

    print(df.a.value_counts())


String Methods
~~~~~~~~~~~~~~~~~~~~~


Merge
------------

Concat
~~~~~~~~~~~~~~~~~~~~~

You can concatenate `Series` and `DataFrames` row-wise.

.. ipython:: python

    print(cudf.concat([s, s]))

    print(cudf.concat([df.head(), df.head()], ignore_index=True))


Join
~~~~~~~~~~~~~~~~~~~~~

You can also do SQL style merges.

.. ipython:: python

    df_a = cudf.DataFrame()
    df_a['key'] = [0, 1, 2, 3, 4]
    df_a['vals_a'] = [float(i + 10) for i in range(5)]

    df_b = cudf.DataFrame()
    df_b['key'] = [1, 2, 4]
    df_b['vals_b'] = [float(i+10) for i in range(3)]

    df_merged = df_a.merge(df_b, on=['key'], how='left')
    print(df_merged.sort_values('key'))


Append
~~~~~~~~~~~~~~~~~~~~~

You can append values from another `Series` or array-like object. Appending `Series` with nulls is not yet supported, but can be done using the `concat` method.

.. ipython:: python

    print(df.a.head().append(df.b.head()))


Grouping
------------

Groupbys involve one or more of the following steps:

    - Splitting the data into groups based on some criteria
    - Applying a function to each group independently
    - Combining the results into a data structure

.. ipython:: python

    df['agg_col1'] = [1 if x % 2 == 0 else 0 for x in range(len(df))]
    df['agg_col2'] = [1 if x % 3 == 0 else 0 for x in range(len(df))]

Grouping and then applying the `sum` function to the resulting groups.


.. ipython:: python

    print(df.groupby('agg_col1').sum())


Grouping hierarchically then applying the `sum` function to the resulting groups.

.. ipython:: python

    print(df.groupby(['agg_col1', 'agg_col2']).sum())


Grouping and applying statistical functions to specific columns, using `agg`.

.. ipython:: python

    print(df.groupby('agg_col1').agg({'a':'max', 'b':'mean', 'c':'sum'}))


Reshaping
------------

Stack
~~~~~~~~~~~~~~~~~~~~~


Pivot Tables
~~~~~~~~~~~~~~~~~~~~~



Time Series
------------
cuDF supports `datetime` typed columns, which allow users to interact with and filter data based on specific timestamps.

.. ipython:: python

    import datetime as dt

    date_df = cudf.DataFrame()
    date_df['date'] = pd.date_range('11/20/2018', periods=72, freq='D')
    date_df['value'] = np.random.sample(len(date_df))

    search_date = dt.datetime.strptime('2018-11-23', '%Y-%m-%d')
    print(date_df.query('date <= @search_date'))


Categoricals
------------

cuDF supports categorical columns.

.. ipython:: python

    pdf = pd.DataFrame({"id":[1,2,3,4,5,6], "grade":['a', 'b', 'b', 'a', 'a', 'e']})
    pdf["grade"] = pdf["grade"].astype("category")

    gdf = cudf.DataFrame.from_pandas(pdf)
    print(gdf)


Accessing the categories of a column.

.. ipython:: python

    print(gdf.grade.cat.categories)

Accessing the underlying code values of each categorical observation.

.. ipython:: python

    print(gdf.grade.cat.codes)

Plotting
------------



Converting Data Representation
--------------------------------


Pandas
~~~~~~~~

Converting a cuDF `DataFrame` to a pandas `DataFrame`.

.. ipython:: python

    print(df.head().to_pandas())

Numpy
~~~~~~~~

Converting a cuDF `DataFrame` to a numpy `rec.array`.

.. ipython:: python

    print(df.to_records())

Converting a cuDF `Series` to a numpy `ndarray`.

.. ipython:: python

    print(df['a'].to_array())

Arrow
~~~~~~~~

Converting a cuDF `DataFrame` to an PyArrow `Table`.

.. ipython:: python

    print(df.to_arrow())


Getting Data In/Out
------------------------


CSV
~~~~

We can write to a CSV file by first sending data to a pandas Dataframe on the host.

.. ipython:: python

    df.to_pandas().to_csv('foo.txt', index=False)


Reading from a csv file. 

.. ipython:: python

    df = cudf.read_csv('foo.txt', delimiter=',',
            names=['a', 'b', 'c'], dtype=['int64', 'int64', 'int64'],
            skiprows=1)
    print(df)


Parquet
~~~~~~~~~


ORC
~~~~~~~~~




Gotchas
--------

