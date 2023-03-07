import numpy as np
from scipy.optimize import shgo
from consts import *
from numpy import array
from Measurements.measurements import get_all_measurements
from Measurements.image import Image
import matplotlib.pyplot as plt
from tmm import coh_tmm
from functions import do_ifft, filtering, do_fft
from sub_eval_tmm_numerical import tmm_eval
from mpl_settings import *


def main(en_plot=True):
    d_s1 = 0.000200

    dir_s1_film = data_dir / "Coated" / sample_names[1]

    measurements = get_all_measurements(data_dir_=dir_s1_film)
    image = Image(measurements)
    s1_film_td = image.get_point(x=eval_point[0], y=eval_point[1], sub_offset=True, add_plot=False)

    #s1_film_td = filtering(s1_film_td, filt_type="hp", wn=0.22, order=1)
    #s1_film_td = filtering(s1_film_td, filt_type="lp", wn=1.55, order=5)
    image.plot_point(x=eval_point[0], y=eval_point[1], y_td=s1_film_td, label="Film")

    s1_film_fd = do_fft(s1_film_td)

    s1_film_ref_td = image.get_ref(both=False, coords=eval_point)

    #s1_film_ref_td = filtering(s1_film_ref_td, filt_type="hp", wn=0.22, order=1)
    #s1_film_ref_td = filtering(s1_film_ref_td, filt_type="lp", wn=1.55, order=5)
    image.plot_point(x=eval_point[0], y=eval_point[1], y_td=s1_film_ref_td, label="Ref.")

    s1_film_ref_fd = do_fft(s1_film_ref_td)

    #plt.show()

    try:
        n_sub = np.load("n_sub.npy")
    except FileNotFoundError:
        n_sub = tmm_eval()
        np.save("n_sub.npy", n_sub)

    freqs = s1_film_ref_fd[:, 0].real
    one = np.ones_like(freqs)
    d_list = [inf, d_sub, d_s1, inf]

    omega = 2 * pi * freqs
    phase_shift = np.exp(-1j * (d_sub + d_s1) * omega / c_thz)

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

    # numerical optimization
    """
    bounds = array([(400, 1200), (400, 1200)])

    def cost(p, f_idx):
        n = array([1, n_sub[f_idx], p[0] + 1j * p[1], 1])
        lam_vac = c_thz / freqs[f_idx]
        t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
        sam_tmm_fd = t_tmm_fd * s1_film_ref_fd[f_idx, 1] * phase_shift[f_idx]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(s1_film_fd[f_idx, 1])) ** 2
        phi_loss = (np.angle(sam_tmm_fd) - np.angle(s1_film_fd[f_idx, 1])) ** 2

        return amp_loss + phi_loss

    n_opt = np.zeros(len(freqs), dtype=complex)
    for f_idx, freq in enumerate(freqs):
        print(f"Frequency: {freq} (THz), (idx: {f_idx})")
        if freq > 2.0:
            n_opt[f_idx] = n_opt[-1]
            continue
        res = shgo(cost, bounds=bounds, args=(f_idx,), iters=6)
        print(res.x, res.fun, "\n")

        n_opt[f_idx] = res.x[0] + 1j * res.x[1]

    n_shgo = array([one, n_sub, n_opt, one], dtype=complex).T

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
    """

    def simple_fit():
        n_line = np.linspace(0, 1200, 1000)

        def cost(p, f_idx):
            n = array([1, n_sub[f_idx], p[0] + 1j * p[1], 1])
            lam_vac = c_thz / freqs[f_idx]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
            sam_tmm_fd = t_tmm_fd * s1_film_ref_fd[f_idx, 1] * phase_shift[f_idx]

            amp_loss = (np.abs(sam_tmm_fd) - np.abs(s1_film_fd[f_idx, 1])) ** 2
            phi_loss = (np.angle(sam_tmm_fd) - np.angle(s1_film_fd[f_idx, 1])) ** 2

            return amp_loss + phi_loss

        n_opt = np.zeros(len(freqs), dtype=complex)
        for f_idx, freq in enumerate(freqs):
            print(f"Frequency: {freq} (THz), (idx: {f_idx})")
            if freq > 2.0:
                n_opt[f_idx] = n_opt[f_idx-1]
                continue
            else:
                best_n, min_val = None, np.inf
                vals = []
                for n in n_line:
                    val = cost([n, n], f_idx)
                    vals.append(val)
                    if val < min_val:
                        best_n = n
                        min_val = val

                n_opt[f_idx] = best_n + 1j * best_n
                print(n_opt[f_idx], "\n")

        return n_opt

    n_opt = simple_fit()

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
        plt.plot(sam_tmm_simple_fd[1:, 0], np.angle(sam_tmm_simple_fd[1:, 1]), label="Simple TMM")

        plt.figure("Time domain")
        plt.plot(sam_tmm_simple_td[:, 0], sam_tmm_simple_td[:, 1], label="Simple TMM")

    def thin_film_model(n_sub, n_film, f_idx):
        omega_i = omega[f_idx]
        t01, t12, t23 = 2 / (1 + n_sub), 2 * n_sub / (n_sub + n_film), 2 * n_film / (n_sub + n_film)
        p1 = np.exp(-1j * omega_i * n_sub * d_sub / c_thz)
        r10, r12, r23 = (n_sub - 1) / (n_sub + 1), (n_sub - n_film) / (n_sub + n_film), (n_film - 1) / (n_film + 1)

        enum = t01 * t12 * t23 * p1
        denum = (1 - p1 ** 2 * r12 * r10) * (1 + r23 * r12)

        return enum / denum

    def simple_model_fit():
        n_line = np.linspace(0, 1200, 1000)

        def cost(p, f_idx):
            n_film = p[1] + 1j * p[1]
            t_func = thin_film_model(n_sub[f_idx], n_film, f_idx) # * phase_shift[f_idx]
            t_func_exp = s1_film_fd[:, 1] / s1_film_ref_fd[:, 1]

            amp_loss = (np.abs(t_func) - np.abs(t_func_exp[f_idx])) ** 2
            phi_loss = (np.angle(t_func) - np.angle(t_func_exp[f_idx])) ** 2

            return amp_loss + phi_loss

        n_opt = np.zeros(len(freqs), dtype=complex)
        for f_idx, freq in enumerate(freqs):
            print(f"Frequency: {freq} (THz), (idx: {f_idx})")
            if freq > 2.0:
                n_opt[f_idx] = n_opt[f_idx - 1]
                continue
            else:
                best_n, min_val = None, np.inf
                vals = []
                for n in n_line:
                    val = cost([n, n], f_idx)
                    vals.append(val)
                    if val < min_val:
                        best_n = n
                        min_val = val

                n_opt[f_idx] = best_n + 1j * best_n
                print(n_opt[f_idx], "\n")

        return n_opt

    n_opt_tmm = n_opt


    """
    n_opt = simple_model_fit()

    n_simple = array([one, n_sub, n_opt, one], dtype=complex).T
    """
    t_func = np.zeros_like(freqs, dtype=complex)
    for f_idx, freq in enumerate(freqs):
        t_func[f_idx] = thin_film_model(n_sub[f_idx], n_opt_tmm[f_idx], f_idx)

    sam_thin_film_fd = t_func * s1_film_ref_fd[:, 1] # * phase_shift
    sam_thin_film_fd = array([freqs, sam_thin_film_fd]).T

    sam_tmm_simple_td = do_fft(sam_thin_film_fd)

    if en_plot:
        plt.figure("RI")
        plt.plot(freqs, n_opt.real, label="Simple model")

        plt.figure("Extinction coefficient")
        plt.plot(freqs, n_opt.imag, label="Simple model")

        plt.figure("Spectrum")
        plt.plot(sam_thin_film_fd[1:, 0], 20 * np.log10(np.abs(sam_thin_film_fd[1:, 1])), label="Simple model")

        plt.figure("Phase")
        plt.plot(sam_thin_film_fd[1:, 0], np.angle(sam_thin_film_fd[1:, 1]), label="Simple model")

        plt.figure("Time domain")
        plt.plot(sam_tmm_simple_td[:, 0], sam_tmm_simple_td[:, 1], label="Simple model")


if __name__ == '__main__':
    main()

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()

    plt.show()

