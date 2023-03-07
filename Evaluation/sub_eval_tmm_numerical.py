import numpy as np
from scipy.optimize import shgo
from consts import *
from numpy import array
from Measurements.measurements import get_all_measurements
from Measurements.image import Image
import matplotlib.pyplot as plt
from tmm import coh_tmm
from functions import do_ifft
from sub_eval import analytical_eval
from mpl_settings import *


def tmm_eval(en_plot=False):
    dir_s1_sub = data_dir / "Uncoated" / sample_names[1]

    measurements = get_all_measurements(data_dir_=dir_s1_sub)
    image = Image(measurements)
    s1_sub_td, s1_sub_fd = image.get_point(x=eval_point[0], y=eval_point[1], sub_offset=True, both=True, add_plot=False)
    s1_sub_ref_td, s1_sub_ref_fd = image.get_ref(both=True, coords=eval_point)

    freqs = s1_sub_fd[:, 0].real
    one = np.ones_like(freqs)

    n_sub = analytical_eval(s1_sub_fd, s1_sub_ref_fd)

    n_list = array([one, n_sub, one]).T
    d_list = [inf, d_sub, inf]

    omega = 2 * pi * freqs
    phase_shift = np.exp(-1j * d_sub * omega / c_thz)

    def calc_model(n_list):
        ts_tmm_fd = np.zeros_like(freqs, dtype=complex)
        for f_idx, freq in enumerate(freqs):
            lam_vac = c_thz / freq
            n = n_list[f_idx]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
            ts_tmm_fd[f_idx] = t_tmm_fd

        sam_tmm_fd = array([freqs, ts_tmm_fd * s1_sub_ref_fd[:, 1] * phase_shift]).T
        sam_tmm_td = do_ifft(sam_tmm_fd)

        return sam_tmm_td, sam_tmm_fd

    sam_tmm_td, sam_tmm_fd = calc_model(n_list)
    if en_plot:
        plt.figure("Spectrum")
        plt.plot(sam_tmm_fd[1:, 0], 20 * np.log10(np.abs(sam_tmm_fd[1:, 1])), label="Analytical TMM")

        plt.figure("Phase")
        plt.plot(sam_tmm_fd[1:, 0], np.angle(sam_tmm_fd[1:, 1]), label="Analytical TMM")

        plt.figure("Time domain")
        plt.plot(sam_tmm_td[:, 0], sam_tmm_td[:, 1], label="Analytical TMM")

    # numerical optimization

    bounds = array([(1.6, 2.1), (0.06, 0.40)])

    def cost(p, f_idx):
        n = array([1, p[0] + 1j * p[1], 1])
        lam_vac = c_thz / freqs[f_idx]
        t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
        sam_tmm_fd = t_tmm_fd * s1_sub_ref_fd[f_idx, 1] * phase_shift[f_idx]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(s1_sub_fd[f_idx, 1])) ** 2
        phi_loss = (np.angle(sam_tmm_fd) - np.angle(s1_sub_fd[f_idx, 1])) ** 2

        return amp_loss + phi_loss

    n_opt = np.zeros(len(freqs))
    for f_idx, freq in enumerate(freqs):
        print(f"Frequency: {freq} (THz), (idx: {f_idx})")
        res = shgo(cost, bounds=bounds, args=(f_idx,), iters=6)
        print(res.x, res.fun, "\n")

        n_opt[f_idx] = res.x[0] + 1j * res.x[1]

    n_shgo = array([one, n_opt, one]).T

    sam_tmm_shgo_td, sam_tmm_shgo_fd = calc_model(n_shgo)

    if en_plot:
        plt.figure("RI")
        plt.plot(freqs, n_opt.real, label="SHGO")

        plt.figure("Extinction coefficient")
        plt.plot(freqs, n_opt.imag, label="SHGO")

        plt.figure("Spectrum")
        plt.plot(sam_tmm_shgo_fd[1:, 0], 20 * np.log10(np.abs(sam_tmm_shgo_fd[1:, 1])), label="SHGO TMM")

        plt.figure("Phase")
        plt.plot(sam_tmm_shgo_fd[1:, 0], np.angle(sam_tmm_shgo_fd[1:, 1]), label="SHGO TMM")

        plt.figure("Time domain")
        plt.plot(sam_tmm_shgo_td[:, 0], sam_tmm_shgo_td[:, 1], label="SHGO TMM")

    return n_opt


if __name__ == '__main__':
    tmm_eval()

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()

    plt.show()


