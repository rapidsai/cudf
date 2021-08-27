Nested
======


Struct and List datatypes
----------------

cuDF supports arbitrarily deep nested lists. Such as ``list(list(int))``, even list of structs or structs of lists

cuDF also supports arbitrary fields for structs - that is, it is possible to have a struct with any number of fields and any number of types that cuDF supports, even a struct of structs

Structs should be made up by same type of datatype or cuDF will produce an error - Example below
    
.. code-block:: python
    
    >>> df = cudf.Series(
    >>> [{'a':'dog', 'b':'cat', 'c':'astronomy'},
    >>> {'a':'fish', 'b':'gerbil', 'c':7}]
    >>> )
    >>> df
        
All rows in a struct column must have the same fields. If a row does not explicitly include a field, the value for that field will be treated as null - Example below

.. code-block:: python

    >>> df = cudf.Series(
    >>> [{'a':'dog', 'b':'cat', 'c':'astronomy'},
    >>> {'a':'fish', 'b':'gerbil'}]
    >>> )
    >>> df
                                          
    0  {'a': 'dog ', 'b': 'cat', 'c': 'astronomy'}
    1      {'a': 'fish', 'b': 'gerbil', 'c': None}
 

