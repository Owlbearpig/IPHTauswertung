import matplotlib.pyplot as plt
import numpy as np


def plot_system_stability(ref_measurements):
    refs = ref_measurements

    plt.figure("System stability")
    t, p2p_ref = [], []
    for i in range(1, len(refs)):
        ref_td = refs[i].get_data_td()
        p2p_ref.append(np.abs(np.max(ref_td[:, 1]) - np.min(ref_td[:, 1])))
        t.append(refs[i].meas_time)
    t0 = sorted(t)[0]
    dt = [(ti-t0).total_seconds() / 60 for ti in t]
    plt.plot(dt, p2p_ref, label="p2p reference")
    plt.ylabel("P2p")
    plt.xlabel("Time (minutes)")
    plt.legend()