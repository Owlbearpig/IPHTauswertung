from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0
from numpy import pi, inf

um = 10**-6
THz = 10 ** 12

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in os.name:
    data_dir = Path(r"/home/alex/Data/IPHT")
else:
    data_dir = Path(r"E:\measurementdata\IPHT")
    try:
        os.scandir(data_dir)
    except FileNotFoundError:
        data_dir = Path(r"C:\Users\Laptop\Desktop")
