import ctypes

# Access the _PyTime_AsSecondsDouble and _PyTime_GetSystemClock functions from pythonapi
clock = ctypes.pythonapi._PyTime_GetSystemClock
as_seconds = ctypes.pythonapi._PyTime_AsSecondsDouble

# Set the argument types and return types of the functions
clock.argtypes = []
clock.restype = ctypes.c_int64

as_seconds.argtypes = [ctypes.c_int64]
as_seconds.restype = ctypes.c_double


