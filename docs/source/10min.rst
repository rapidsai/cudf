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

Creating a `Series`

.. ipython:: python

  s = cudf.Series([1,2,3,None,4])
  print(s)

Creating a `Dataframe` from a list of tuples.

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

Examples of how to view the top rows of the GPU dataframe:

.. ipython:: python

    print(df.head(2))

Sorting by values:

.. ipython:: python

    print(df.sort_values(by='a', ascending=False))


Selection
------------

Getting
~~~~~~~~~~~~~~

Selecting a single column, which yields a `cudf.Series`, equivalent to `df.a`:

.. ipython:: python

    print(df['a'])



Selection by Label
~~~~~~~~~~~~~~~~~~~~~

Select rows from index 2 to index 5 from columns 'a' and 'b'.

.. ipython:: python

    print(df.loc[2:5, ['a', 'b']])



Selection by Position
~~~~~~~~~~~~~~~~~~~~~

Boolean Indexing
~~~~~~~~~~~~~~~~~~~~~

Setting
~~~~~~~~~~~~~~~~~~~~~


Missing Data
------------


Operations
------------

Stats
~~~~~~~~~~~~~~~~~~~~~

Applymap
~~~~~~~~~~~~~~~~~~~~~

Applying functions to a `Series`:

.. ipython:: python

    def add_ten(num):
        return num + 10

    print(df['a'].applymap(add_ten))


Histogramming
~~~~~~~~~~~~~~~~~~~~~


String Methods
~~~~~~~~~~~~~~~~~~~~~


Merge
------------

Concat
~~~~~~~~~~~~~~~~~~~~~


Join
~~~~~~~~~~~~~~~~~~~~~


Append
~~~~~~~~~~~~~~~~~~~~~


Grouping
------------



Reshaping
------------

Stack
~~~~~~~~~~~~~~~~~~~~~


Pivot Tables
~~~~~~~~~~~~~~~~~~~~~



Time Series
------------


Categoricals
------------


Plotting
------------


Getting Data In/Out
------------


CSV
~~~~


We can write to a CSV file by first sending data to a pandas Dataframe on the host.

.. ipython:: python

    df.to_pandas().to_csv('foo.txt', index=False)


Reading from a csv file. `read_csv` requires explicit types, and can accept different delimiters and line terminators.


.. ipython:: python

    df = cudf.read_csv('foo.txt', delimiter=',',
            names=['a', 'b', 'c'], dtype=['int64', 'int64', 'int64'],
            skiprows=1)
    print(df)

HDF5
~~~~~~~~~


Excel
~~~~~~~~~



Gotchas
--------