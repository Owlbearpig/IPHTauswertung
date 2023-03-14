import random

import numpy as np
from scipy.optimize import shgo
from consts import *
from numpy import array
import matplotlib.pyplot as plt
from tmm import coh_tmm
from functions import do_ifft, do_fft, phase_correction, to_db, window
from sub_eval import analytical_eval
from mpl_settings import *


def tmm_eval(sub_image, eval_point, en_plot=False, analytical=False, single_f_idx=None):
    sam_idx = sub_image.sample_idx
    sub_td = sub_image.get_point(x=eval_point[0], y=eval_point[1], sub_offset=True, both=False)
    sub_ref_td = sub_image.get_ref(both=False, coords=eval_point)

    sub_td = window(sub_td, win_len=12, shift=0, en_plot=False)
    sub_ref_td = window(sub_ref_td, win_len=12, shift=0, en_plot=False)

    sub_ref_fd, sub_fd = do_fft(sub_ref_td), do_fft(sub_td)

    sub_ref_fd = phase_correction(sub_ref_fd, fit_range=(0.55, 1.00), extrapolate=True, ret_fd=True)
    sub_fd = phase_correction(sub_fd, fit_range=(0.55, 1.00), extrapolate=True, ret_fd=True)

    freqs = sub_fd[:, 0].real
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

        sam_tmm_fd = array([freqs, ts_tmm_fd * sub_ref_fd[:, 1] * phase_shift]).T
        sam_tmm_td = do_ifft(sam_tmm_fd)

        return sam_tmm_td, sam_tmm_fd

    if analytical:
        n_sub = analytical_eval(sub_fd, sub_ref_fd)
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
            n_sub = np.load(f"n_sub_s{sam_idx + 1}_{eval_point[0]}_{eval_point[1]}.npy")
        except FileNotFoundError:
            # numerical optimization
            bounds = shgo_bounds_sub[sam_idx]

            def cost(p, f_idx):
                n = array([1, p[0] + 1j * p[1], 1])
                lam_vac = c_thz / freqs[f_idx]
                t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
                sam_tmm_fd = t_tmm_fd * sub_ref_fd[f_idx, 1] * phase_shift[f_idx]

                amp_loss = (np.abs(sam_tmm_fd) - np.abs(sub_fd[f_idx, 1])) ** 2
                phi_loss = (np.angle(sam_tmm_fd) - np.angle(sub_fd[f_idx, 1])) ** 2

                return amp_loss + phi_loss

            if single_f_idx is not None:
                freq = freqs[single_f_idx]
                print(f"Frequency: {freq} (THz), (idx: {single_f_idx})")
                res = shgo(cost, bounds=bounds, args=(single_f_idx,), iters=shgo_iters)
                print(res.x, res.fun)

                return res.x[0] + 1j * res.x[1]
            else:
                n_sub = np.zeros(len(freqs), dtype=complex)
                for f_idx, freq in enumerate(freqs):
                    # lol while loop maybe ???
                    if freq <= 4:
                        print(f"Frequency: {freq} (THz), (idx: {f_idx})")
                        if freq <= 0.25:
                            res = shgo(cost, bounds=bounds, args=(f_idx,), iters=5)
                        else:
                            res = shgo(cost, bounds=bounds, args=(f_idx,), iters=shgo_iters-2)
                            if res.fun > 1e-5:
                                res = shgo(cost, bounds=bounds, args=(f_idx,), iters=shgo_iters)
                                if res.fun > 1e-5:
                                    res = shgo(cost, bounds=bounds, args=(f_idx,), iters=shgo_iters + 1)
                                    if res.fun > 1e-5:
                                        res = shgo(cost, bounds=bounds, args=(f_idx,), iters=shgo_iters + 2)

                        print(res.x, res.fun, "\n")

                        n_sub[f_idx] = res.x[0] + 1j * res.x[1]
                    else:
                        n_sub[f_idx] = n_sub[f_idx-1]

        n_shgo = array([one, n_sub, one]).T

        sam_tmm_shgo_td, sam_tmm_shgo_fd = calc_model(n_shgo)

        phi_tmm = phase_correction(sam_tmm_shgo_fd, disable=False, fit_range=(0.55, 1.0))
        label = f"{sub_image.name} (TMM) x={eval_point[0]} mm, y={eval_point[1]} mm"
        if en_plot:
            plt.figure("RI")
            plt.title("Refractive index substrates")
            plt.plot(freqs[plot_range_sub], n_sub[plot_range_sub].real, label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Refractive index")

            plt.figure("Extinction coefficient")
            plt.title("Extinction coefficient substrates")
            plt.plot(freqs[plot_range_sub], n_sub[plot_range_sub].imag, label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Extinction coefficient")

            plt.figure("Spectrum")
            plt.title("Spectrum substrates")
            plt.plot(sam_tmm_shgo_fd[plot_range1, 0], to_db(sam_tmm_shgo_fd[plot_range1, 1]), label=label, zorder=2,
                     color="Green")

            plt.figure("Phase")
            plt.title("Phases substrates")
            plt.plot(sam_tmm_shgo_fd[plot_range_sub, 0], phi_tmm[plot_range_sub, 1],
                     label=label, linewidth=1.5)

            plt.figure("Time domain")
            plt.plot(sam_tmm_shgo_td[:, 0], sam_tmm_shgo_td[:, 1], label=label, linewidth=2)

    return n_sub


if __name__ == '__main__':
    from Measurements.image import Image

    sam_idx = 3
    image_data = data_dir / "Uncoated" / sample_names[sam_idx]
    image = Image(image_data)

    eval_point = random.choice(image.all_points)
    # eval_point = (20, 10)  # used for s1-s3
    # eval_point = (33, 11)  # s4

    image.plot_point(*eval_point, label=f"Sample {sam_idx+1} Uncoated")
    n_sub = tmm_eval(sub_image=image, eval_point=eval_point, en_plot=True)
    np.save(f"n_sub_s{sam_idx + 1}_{eval_point[0]}_{eval_point[1]}.npy", n_sub)

    """
    sam_idx = 1
    image_data = data_dir / "Uncoated" / sample_names[sam_idx]
    eval_point = (20, 9)  # used for s1-s3
    # eval_point = (33, 11)  # s4

    image = Image(image_data)
    image.plotted_ref = True
    image.plot_point(*eval_point, label=f"Sample {sam_idx + 1} Uncoated")
    n_sub = tmm_eval(sub_image=image, eval_point=eval_point, en_plot=True)
    np.save(f"n_sub_s{sam_idx + 1}_{eval_point[0]}_{eval_point[1]}.npy", n_sub)
    """

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()

    plt.show()
