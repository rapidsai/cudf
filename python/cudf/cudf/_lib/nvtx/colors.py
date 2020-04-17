# Copyright (c) 2020, NVIDIA CORPORATION.

import functools

_NVTX_COLORS = {
    "green": 0x008000,
    "blue": 0x0000FF,
    "yellow": 0xFFFF00,
    "purple": 0x800080,
    "rapids": 0x7400FF,
    "cyan": 0x00FFFF,
    "red": 0xFF0000,
    "white": 0xFFFFFF,
    "darkgreen": 0x006400,
    "orange": 0xFFA500,
}


@functools.lru_cache()
def color_to_hex(color="blue"):
    """
    Convert color to ARGB hex value.
    """
    if color in _NVTX_COLORS:
        return _NVTX_COLORS[color]
    try:
        import matplotlib.colors
    except ImportError as e:
        raise TypeError(
            f"Invalid color {color}. Please install matplotlib "
            "for additional colors support"
        ) from e
    rgba = matplotlib.colors.to_rgba(color)
    argb = (rgba[-1], rgba[0], rgba[1], rgba[2])
    return int(matplotlib.colors.to_hex(argb, keep_alpha=True)[1:], 16)
