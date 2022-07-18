# Options

cuDF has an options API to configure and customize global behavior.
This API complements the [pandas.options](https://pandas.pydata.org/docs/user_guide/options.html) API with features specific to cuDF.

User may get the full list of cudf options with {py:func}`cudf.describe_option` with no arguments.
To set value to a option, use {py:func}`cudf.set_option`.

See [API reference](api.options) for detail.
