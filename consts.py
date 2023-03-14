from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0
from scipy.constants import epsilon_0
from numpy import pi, inf, array

# Directly indexed; don't change order
sample_names = ["10x10cm_sqrd_s1", "10x10cm_sqrd_s2", "10x10cm_sqrd_s3", "5x5cm_sqrd"]
sample_labels = ["(200 nm Ag)", "(500 nm AlZnO)", "(200 nm Ag + 500 nm AlZnO)", "(200 nm ITO)"]
sample_thicknesses = [0.0002, 0.0005, 0.0007, 0.0002]  # in mm
shgo_bounds = [[(0, 800), (0, 800)],
               [(0, 30), (0, 30)],
               [(100, 200), (0, 100)],
               [(0, 300), (0, 100)],
               ]
sub_bounds = [[(1.5, 2.1), (0.01, 0.30)],
              [(1.5, 2.1), (0.01, 0.30)],
              [(1.5, 2.1), (0.01, 0.30)],
              [(1.5, 2.1), (0.01, 0.30)],
              ]

drude_bounds = [[(0, 1e7), (0, 1)],
                [(0, 3e5), (0, 1)],
                [(0, 1e7), (0, 1)],
                [(0, 1e6), (0, 1)],
                ]
td_scales = [50, 1, 50, 10]

shgo_iters = 7
d_sub = 0.070  # mm
angle_in = 0 * pi / 180

plot_range = slice(25, 200)
plot_range1 = slice(1, 500)
plot_range_sub = slice(25, 350)
# eval_point = (10, 10)#(20, 9)

cur_os = os.name

c_thz = c0 * 10 ** -9

um = 10 ** -6
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
