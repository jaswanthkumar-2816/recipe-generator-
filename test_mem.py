import psutil
import os

print("Initial RAM:", psutil.Process().memory_info().rss / 1024 / 1024, "MB")

from Foodimg2Ing.output import _get_assets
print("Imports loaded RAM:", psutil.Process().memory_info().rss / 1024 / 1024, "MB")

_get_assets()
print("Assets loaded RAM:", psutil.Process().memory_info().rss / 1024 / 1024, "MB")
