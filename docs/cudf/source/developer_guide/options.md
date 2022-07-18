# Options

Options are stored as a dictionary in `cudf.options` module.
Each option name is also its key in the dictionary.
The value of the option is an instance of a `CUDFOption` object.

A `CUDFOption` object has the following attributes:
- `value`: the current value of the option
- `description`: a text description of the option
- `validator`: a boolean function that returns `True` if value is valid,
`False` otherwise.

Developers can use `cudf.options._register_option` to add options to the dictionary.
`cudf.get_option` is provided to get config value from the dictionary.

When testing the behavior of a certain option,
it is advised to use [yield fixture](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#yield-fixtures-recommended) to setup and cleanup the option.

See [API reference](api.options) for detail.
