# CUDF Configuration

Configurations are stored as a dictionary in `cudf.config` module.
Each configuration name is also its key in the dictionary.
The value of the configuration is an instance of a `CUDFConfiguration` object.

A `CUDFConfiguration` object inherits from `dataclass` and consists 4 attributes:
- `name`: the name and the key of the configuration
- `value`: the current value of the configuration
- `description`: a text description of the configuration
- `validator`: a boolean function that returns `True` if value is valid,
`False` otherwise.

Developers can use `cudf.register_config` to add configurations to the registry.
`cudf.get_config` is provided to get config value from the registry.

When testing the behavior of certain configuration,
it is advised to use [yield fixture](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#yield-fixtures-recommended) to setup and cleanup certain configuration for the test.

See [API reference](api.config) for detail.
