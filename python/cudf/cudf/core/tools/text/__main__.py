import importlib.resources
import os
import sys

# Register cugrep as a CLI if run as a module

def main():
    from .cugrep import cugrep
    cugrep()

if __name__ == "__main__":
    main()
