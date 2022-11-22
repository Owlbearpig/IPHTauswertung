import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linspace, inf, nan_to_num, sum, zeros
from consts import THz, c0, pi, um, ROOT_DIR

#print(mpl.rcParams.keys())

# mpl.rcParams['lines.linestyle'] = '--'
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['axes.grid'] = True
# plt.style.use(['dark_background'])
# plt.xkcd()

mpl.rcParams.update({'font.size': 22})

post_process_config = {"sub_offset": True, "en_windowing": False}

verbose = False
