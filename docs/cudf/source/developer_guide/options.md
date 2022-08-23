# Options

The usage of options is explained in the [user guide](options_user_guide).
This document provides more explanations on how developers work with options internally.

Options are stored as a dictionary in the `cudf.options` module.
Each option name is its key in the dictionary.
The value of the option is an instance of an `Options` object.

An `Options` object has the following attributes:
- `value`: the current value of the option
- `description`: a text description of the option
- `validator`: a boolean function that returns `True` if `value` is valid,
`False` otherwise.

Developers can use `cudf.options._register_option` to add options to the dictionary.
{py:func}`cudf.get_option` is provided to get option values from the dictionary.

When testing the behavior of a certain option,
it is advised to use [`yield` fixture](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#yield-fixtures-recommended) to set up and clean up the option.

See the [API reference](api.options) for more details.
