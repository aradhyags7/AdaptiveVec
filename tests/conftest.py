import os
import sys

# On Windows, register MinGW UCRT64 runtime DLLs for C++ pybind11 modules
if sys.platform == "win32":
    mingw_bin = r"C:\msys64\ucrt64\bin"
    if os.path.exists(mingw_bin) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(mingw_bin)
        except OSError:
            pass
