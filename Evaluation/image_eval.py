import numpy as np
from scipy.optimize import shgo
from consts import *
from numpy import array
from Measurements.measurements import get_all_measurements
from Measurements.image import Image
import matplotlib.pyplot as plt
from tmm import coh_tmm
from functions import do_ifft, filtering, do_fft, phase_correction
from sub_eval_tmm_numerical import tmm_eval
from mpl_settings import *




def main(en_plot=True):
    d_film = 0.000200
    plot_td_scale = 50

    dir_s1_film = data_dir / "Coated" / sample_names[1]

    measurements = get_all_measurements(data_dir_=dir_s1_film)
    image = Image(measurements)
    image.plot_image()

    s1_film_td = image.get_point(x=eval_point[0], y=eval_point[1], sub_offset=True)

    s1_film_fd = do_fft(s1_film_td)

    s1_film_ref_td = image.get_ref(both=False, coords=eval_point)

    s1_film_ref_fd = do_fft(s1_film_ref_td)

    #plt.show()

    try:
        n_sub = np.load("n_sub.npy")
    except FileNotFoundError:
        n_sub = tmm_eval()
        np.save("n_sub.npy", n_sub)

    freqs = s1_film_ref_fd[:, 0].real
    one = np.ones_like(freqs)
    d_list = [inf, d_sub, d_film, inf]

    omega = 2 * pi * freqs
    phase_shift = np.exp(-1j * (d_sub + d_film) * omega / c_thz)

    def calc_model(n_list):
        ts_tmm_fd = np.zeros_like(freqs, dtype=complex)
        for f_idx, freq in enumerate(freqs):
            lam_vac = c_thz / freq
            n = n_list[f_idx]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
            ts_tmm_fd[f_idx] = t_tmm_fd

        sam_tmm_fd = array([freqs, ts_tmm_fd * s1_film_ref_fd[:, 1] * phase_shift]).T
        sam_tmm_td = do_ifft(sam_tmm_fd)

        return sam_tmm_td, sam_tmm_fd

    def simple_fit(selected_freq=0.800):




        n_opt = np.zeros(len(freqs), dtype=complex)
        for f_idx, freq in enumerate(freqs):
            if freq < 2.0:
                print(f"Frequency: {freq} (THz), (idx: {f_idx})")
                res = shgo(cost, bounds=bounds, args=(f_idx, ), iters=7)
                """
                best_n, min_val = None, np.inf
                vals = []
                for n in n_line:
                    val = cost([n, n], f_idx)
                    vals.append(val)
                    if val < min_val:
                        best_n = n
                        min_val = val
                n_opt[f_idx] = best_n + 1j * best_n
                """
                n_opt[f_idx] = res.x[0] + 1j * res.x[1]
                print(n_opt[f_idx], "\n")
            else:
                n_opt[f_idx] = n_opt[f_idx - 1]
                continue

        return n_opt

    try:
        n_opt = np.load("n_opt_simple_fit.npy")
    except FileNotFoundError:
        n_opt = simple_fit()
        np.save("n_opt_simple_fit.npy", n_opt)

    epsilon = n_opt ** 2
    sigma = 1j * (1 - epsilon) * epsilon_0 * omega * THz

    plt.figure("Conductivity")
    plt.plot(freqs, sigma.real, label="Sigma real part")
    plt.plot(freqs, sigma.imag, label="Sigma imag part")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Conductivity (S/m)")

    n_simple = array([one, n_sub, n_opt, one], dtype=complex).T

    sam_tmm_simple_td, sam_tmm_simple_fd = calc_model(n_simple)

    if en_plot:
        plt.figure("RI")
        plt.plot(freqs, n_opt.real, label="Simple")

        plt.figure("Extinction coefficient")
        plt.plot(freqs, n_opt.imag, label="Simple")

        plt.figure("Spectrum")
        plt.plot(sam_tmm_simple_fd[1:, 0], 20 * np.log10(np.abs(sam_tmm_simple_fd[1:, 1])), label="Simple TMM")

        plt.figure("Phase")
        plt.plot(sam_tmm_simple_fd[1:, 0], phase_correction(sam_tmm_simple_fd[1:, ]), label="Simple TMM")

        plt.figure("Time domain")
        plt.plot(sam_tmm_simple_td[:, 0], plot_td_scale*sam_tmm_simple_td[:, 1], label="Simple TMM")





if __name__ == '__main__':
    main()

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()

    plt.show()

