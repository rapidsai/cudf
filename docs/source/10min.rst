10 Minutes to cuDF
=======================

Modeled after 10 Minutes to Pandas, this is a short introduction to cuDF, geared mainly for new users.

.. ipython:: python
   :suppress:

   import os
   import cudf
   import numpy as np
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

.. ipython:: python

    gdf = cudf.DataFrame({
    'a':[1,2,3],
    'b':[4,5,6],
    'c':[7,8,9]
    })
    print(gdf)

test

Viewing Data
-------------


Selection
------------

Getting
~~~~~~~~~~~~~~

Selection by Label
~~~~~~~~~~~~~~~~~~~~~


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

Apply
~~~~~~~~~~~~~~~~~~~~~

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
Coming in a future release.




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