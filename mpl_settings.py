import matplotlib as mpl
from consts import cur_os, Path
import matplotlib.pyplot as plt
#print(mpl.rcParams.keys())

# mpl.rcParams['lines.linestyle'] = '--'
#mpl.rcParams['legend.fontsize'] = 'large' #'x-large'
mpl.rcParams['legend.shadow'] = False
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['lines.markersize'] = 4
mpl.rcParams['lines.linewidth'] = 3.5 #2
mpl.rcParams['ytick.major.width'] = 2.5
mpl.rcParams['xtick.major.width'] = 2.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams.update({'font.size': 24})

if 'posix' in cur_os:
    result_dir = Path(r"/home/alex/MEGA/AG/Projects/THz Conductivity/IPHT/Results")
else:
    result_dir = Path(r"E:\Mega\AG\Projects\THz Conductivity\IPHT\Results")
mpl.rcParams["savefig.directory"] = result_dir


