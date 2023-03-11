import numpy as np
from scipy.optimize import shgo
from consts import *
from numpy import array
import matplotlib.pyplot as plt
from tmm import coh_tmm
from functions import do_ifft
from sub_eval import analytical_eval
from mpl_settings import *


def tmm_eval(sub_image, meas_point, en_plot=False, analytical=False, single_f_idx=None):
    s1_sub_td, s1_sub_fd = sub_image.get_point(x=meas_point[0], y=meas_point[1], sub_offset=True, both=True)
    s1_sub_ref_td, s1_sub_ref_fd = sub_image.get_ref(both=True, coords=meas_point)

    freqs = s1_sub_fd[:, 0].real
    omega = 2 * pi * freqs
    one = np.ones_like(freqs)
    phase_shift = np.exp(-1j * d_sub * omega / c_thz)

    d_list = [inf, d_sub, inf]

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

    if analytical:
        n_sub = analytical_eval(s1_sub_fd, s1_sub_ref_fd)
        if single_f_idx is not None:
            return n_sub[single_f_idx]

        n_list = array([one, n_sub, one]).T

        sam_tmm_td, sam_tmm_fd = calc_model(n_list)
        if en_plot:
            plt.figure("Spectrum")
            plt.plot(sam_tmm_fd[1:, 0], 20 * np.log10(np.abs(sam_tmm_fd[1:, 1])), label="Analytical TMM")

            plt.figure("Phase")
            plt.plot(sam_tmm_fd[1:, 0], np.angle(sam_tmm_fd[1:, 1]), label="Analytical TMM")

            plt.figure("Time domain")
            plt.plot(sam_tmm_td[:, 0], sam_tmm_td[:, 1], label="Analytical TMM")
    else:
        try:
            n_sub = np.load("n_sub_s{sam}.npy")
        except FileNotFoundError:
            # numerical optimization
            bounds = array([(1.6, 2.1), (0.01, 0.40)])

            def cost(p, f_idx):
                n = array([1, p[0] + 1j * p[1], 1])
                lam_vac = c_thz / freqs[f_idx]
                t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
                sam_tmm_fd = t_tmm_fd * s1_sub_ref_fd[f_idx, 1] * phase_shift[f_idx]

                amp_loss = (np.abs(sam_tmm_fd) - np.abs(s1_sub_fd[f_idx, 1])) ** 2
                phi_loss = (np.angle(sam_tmm_fd) - np.angle(s1_sub_fd[f_idx, 1])) ** 2

                return amp_loss + phi_loss

            if single_f_idx is not None:
                freq = freqs[single_f_idx]
                print(f"Frequency: {freq} (THz), (idx: {single_f_idx})")
                res = shgo(cost, bounds=bounds, args=(single_f_idx,), iters=7)
                print(res.x, res.fun)

                return res.x[0] + 1j * res.x[1]
            else:
                n_sub = np.zeros(len(freqs), dtype=complex)
                for f_idx, freq in enumerate(freqs):
                    print(f"Frequency: {freq} (THz), (idx: {f_idx})")
                    res = shgo(cost, bounds=bounds, args=(f_idx,), iters=7)
                    print(res.x, res.fun, "\n")

                    n_sub[f_idx] = res.x[0] + 1j * res.x[1]

        n_shgo = array([one, n_sub, one]).T

        sam_tmm_shgo_td, sam_tmm_shgo_fd = calc_model(n_shgo)

        if en_plot:
            plt.figure("RI")
            plt.plot(freqs, n_sub.real, label="SHGO")

            plt.figure("Extinction coefficient")
            plt.plot(freqs, n_sub.imag, label="SHGO")

            plt.figure("Spectrum")
            plt.plot(sam_tmm_shgo_fd[1:, 0], 20 * np.log10(np.abs(sam_tmm_shgo_fd[1:, 1])), label="SHGO TMM")

            plt.figure("Phase")
            plt.plot(sam_tmm_shgo_fd[1:, 0], np.angle(sam_tmm_shgo_fd[1:, 1]), label="SHGO TMM")

            plt.figure("Time domain")
            plt.plot(sam_tmm_shgo_td[:, 0], sam_tmm_shgo_td[:, 1], label="SHGO TMM")

    return n_sub


if __name__ == '__main__':
    from Measurements.image import Image

    image_data = data_dir / "Uncoated" / sample_names[0]
    image = Image(image_data)
    image.plot_point(20, 9)
    n_sub = tmm_eval(sub_image=image, meas_point=(20, 9), en_plot=True)
    np.save("n_sub.npy", n_sub)

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()

    plt.show()


