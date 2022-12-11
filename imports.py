import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linspace, inf, nan_to_num, sum, zeros
from consts import THz, c0, pi, um, ROOT_DIR, cur_os, Path

#print(mpl.rcParams.keys())

# mpl.rcParams['lines.linestyle'] = '--'
#mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['savefig.format'] = 'pdf'
if 'posix' in cur_os:
    result_dir = Path(r"/home/alex/MEGA/AG/Projects/THz Conductivity/IPHT/5x5mm_sqrd/results")
else:
    result_dir = Path(r"E:\Mega\AG\Projects\THz Conductivity\IPHT\5x5mm_sqrd\results")
mpl.rcParams["savefig.directory"] = result_dir
mpl.rcParams.update({'font.size': 22})
# plt.style.use(['dark_background'])
# plt.xkcd()

post_process_config = {"sub_offset": True, "en_windowing": False}

verbose = False
