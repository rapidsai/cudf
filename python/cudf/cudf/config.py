_CUDF_CONFIG = {"binary_operation_result_type": "PROMOTE"}

def get_config(key):
    return _CUDF_CONFIG[key]

def set_config(key, val):
    if key not in _CUDF_CONFIG:
        raise ValueError(f"Unrecognized key for cudf configs: {key}")
    _CUDF_CONFIG[key] = val
