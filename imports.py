import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linspace, inf, nan_to_num, sum, zeros
from consts import THz, c0, pi, um, ROOT_DIR, cur_os, Path
from mpl_settings import *

post_process_config = {"sub_offset": True, "en_windowing": False}

verbose = False
