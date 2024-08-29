""" CPU-time returning clock() function which works from within njit-ted code """
import ctypes
from ctypes.util import find_library

__LIB = find_library("c")

clock = ctypes.CDLL(__LIB).clock
clock.argtypes = []
