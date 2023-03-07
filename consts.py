from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0
from numpy import pi, inf

sample_names = ["5x5cm_sqrd", "10x10cm_sqrd_s1", "10x10cm_sqrd_s2", "10x10cm_sqrd_s3"]

d_sub = 0.070  # mm
angle_in = 0 * pi / 180

eval_point = (19, 9)

cur_os = os.name

c_thz = c0 * 10 ** -9

um = 10**-6
THz = 10 ** 12

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in cur_os:
    data_dir = Path(r"/home/alex/Data/IPHT")
else:
    data_dir = Path(r"E:\measurementdata\IPHT")
    try:
        os.scandir(data_dir)
    except FileNotFoundError:
        data_dir = Path(r"C:\Users\Laptop\Desktop")
