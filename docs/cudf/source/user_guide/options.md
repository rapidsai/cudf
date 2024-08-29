(options_user_guide)=

# Options

cuDF has an options API to configure and customize global behavior.
This API complements the [pandas.options](https://pandas.pydata.org/docs/user_guide/options.html) API with features specific to cuDF.

{py:func}`cudf.describe_option` will print the option's description,
the current value, and the default value.
When no argument is provided,
all options are printed.
To set value to a option, use {py:func}`cudf.set_option`.

See the [options API reference](api.options) for descriptions of the available options.
