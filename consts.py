from pathlib import Path
import os
import numpy as np
from scipy.constants import c as c0
from scipy.constants import epsilon_0
from numpy import pi, inf, array

# Directly indexed; don't change order
sample_names = ["10x10cm_sqrd_s1", "10x10cm_sqrd_s2", "10x10cm_sqrd_s3", "5x5cm_sqrd", "2023-03-20", "2023-03-21"]
sample_labels = ["(200 nm Ag)", "(500 nm Al:ZnO)", "(200 nm Ag + 500 nm Al:ZnO)", "(200 nm ITO)"]
# sample_labels = ["(200 nm Ag)", "(500 nm Al:ZnO)", "(200 nm Ag)", "(200 nm ITO)"] # layer 1
# sample_labels = ["(200 nm Ag)", "(500 nm Al:ZnO)", "(500 nm Al:ZnO)", "(200 nm ITO)"] # layer 2
# sample_thicknesses = [0.0002, 0.0005, [0.0002, 0.0005], 0.0002]  # in mm
sample_thicknesses = [0.0002, 0.0005, 0.0007, 0.0002]  # in mm
shgo_bounds_film = [[(1, 600), (0, 600)],
                    [(5, 27), (10, 27)],
                    # [(50, 300), (50, 300), (5, 70), (10, 75)],
                    [(1, 200), (0, 200)],
                    # [(1, 700), (1, 100), (1, 60)],
                    [(1, 700), (0, 80), ],
                    [(30, 100), (30, 90)],
                    ]

shgo_bounds_sub = [[(1.5, 1.9), (0.01, 0.30)],
                   [(1.5, 1.9), (0.01, 0.30)],
                   [(1.5, 1.9), (0.01, 0.30)],
                   [(1.5, 1.9), (0.01, 0.40)],
                   ]

shgo_bounds_drude = [[(0, 1e7), (0, 1)],
                     [(0, 3e5), (0, 1)],
                     [(0, 1e7), (0, 1)],
                     [(0, 1e6), (0, 1)],
                     ]
td_scales = [50, 1, 50, 10]

shgo_iters = 7
d_sub = 0.070  # mm
angle_in = 0 * pi / 180
drude_fit_range = (0.3, 2.0)

plot_range = slice(25, 200)
# plot_range = slice(25, 1000)
plot_range1 = slice(0, 1000)
# plot_range1 = slice(1, 1000)
plot_range_sub = slice(25, 350)
# plot_range_sub = slice(25, 1000)
# eval_point = (10, 10)#(20, 9)

cur_os = os.name

c_thz = c0 * 10 ** -9  # mm / s

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
