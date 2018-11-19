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

   #### portions of this were borrowed from the
   #### cuDF cheatsheet and existing docs
   #### created November, 2018
   #### Nick Becker (NVIDIA), 

This is a basic example


.. ipython:: python

    x = 2
    x**3


Object Creation
---------------

Series

.. ipython:: python

  s = cudf.Series([1,2,3,None,4])
  print(s)

Dataframe from dictionary

.. ipython:: python

    df = cudf.DataFrame([('a', list(range(20))),
    ('b', list(reversed(range(20)))),
    ('c', list(range(20)))])
    print(df)

Dataframe from pandas 

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
.. ipython:: python

    # get rows from index 2 to index 5 from 'a' and 'b' columns.
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


HDF5
~~~~~~~~~


Excel
~~~~~~~~~



Gotchas
--------