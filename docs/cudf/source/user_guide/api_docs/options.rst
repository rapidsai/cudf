.. _api.options:

====================
Options and settings
====================

.. autosummary::
   :toctree: api/

   cudf.get_option
   cudf.set_option
   cudf.describe_option
   cudf.option_context


Available options
-----------------

You can get a list of available options and their descriptions with :func:`~cudf.describe_option`. When called
with no argument :func:`~cudf.describe_option` will print out the descriptions for all available options.

.. ipython:: python

   import cudf
   cudf.describe_option()
