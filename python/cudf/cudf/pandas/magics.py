# Copyright (c) 2022-2023, NVIDIA CORPORATION.


def load_ipython_extension(ip):
    import xdf.autoload
    from xdf.magics import XDFMagics

    xdf.autoload.install()
    ip.register_magics(XDFMagics)
