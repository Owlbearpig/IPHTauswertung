import numpy as np
from scipy.optimize import shgo
from consts import *
from numpy import array
from Measurements.measurements import get_all_measurements
from Measurements.image import Image
import matplotlib.pyplot as plt
from tmm import coh_tmm
from functions import do_ifft, filtering, do_fft, phase_correction, to_db, get_noise_floor
from sub_eval_tmm_numerical import tmm_eval
from mpl_settings import *


def main(en_plot=True):
    d_film = 0.000200
    plot_td_scale = 50
    eval_point = (30.0, 10.0)
    sample_idx = 0
    plt_title = f"Sample {sample_idx + 1} (200 nm Ag)"

    dir_s1_film = data_dir / "Coated" / sample_names[sample_idx]

    image = Image(dir_s1_film)
    s1_film_td = image.get_point(x=eval_point[0], y=eval_point[1], sub_offset=True, add_plot=False)

    # s1_film_td = filtering(s1_film_td, filt_type="hp", wn=0.25, order=5)
    # s1_film_td = filtering(s1_film_td, filt_type="lp", wn=1.75, order=5)
    image.plot_point(x=eval_point[0], y=eval_point[1], y_td=s1_film_td, label=f"Sample", td_scale=plot_td_scale,
                     sub_noise_floor=True)

    s1_film_fd = do_fft(s1_film_td)

    s1_film_ref_td = image.get_ref(both=False, coords=eval_point)

    # s1_film_ref_td = filtering(s1_film_ref_td, filt_type="hp", wn=0.25, order=5)
    # s1_film_ref_td = filtering(s1_film_ref_td, filt_type="lp", wn=1.75, order=5)
    # image.plot_point(x=eval_point[0], y=eval_point[1], y_td=s1_film_ref_td, label="Reference")

    s1_film_ref_fd = do_fft(s1_film_ref_td)

    # plt.show()

    dir_s1_film = data_dir / "Uncoated" / sample_names[sample_idx]
    image_sub = Image(dir_s1_film)

    try:
        n_sub = np.load("n_sub.npy")
    except FileNotFoundError:
        n_sub = tmm_eval(image_sub, meas_point=(20, 9))
        np.save("n_sub.npy", n_sub)

    # n_sub *= np.random.random(n_sub.shape)

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
        n_line = np.linspace(100, 250, 1000)
        bounds = [(0, 800), (0, 800)]
        #phi_corrected = phase_correction(s1_film_fd, ret_interpol=True, fit_range=[0.35, 1.65])

        #phi_corrected = np.angle(np.exp(1j * phi_corrected))

        def cost(p, f_idx):
            n = array([1, n_sub[f_idx], p[0] + 1j * p[1], 1])
            lam_vac = c_thz / freqs[f_idx]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
            sam_tmm_fd = t_tmm_fd * s1_film_ref_fd[f_idx, 1] * phase_shift[f_idx]

            amp_loss = (np.abs(sam_tmm_fd) - np.abs(s1_film_fd[f_idx, 1])) ** 2
            # phi_loss = (np.angle(sam_tmm_fd) - phi_corrected[f_idx]) ** 2
            phi_loss = (np.angle(sam_tmm_fd) - np.angle(s1_film_fd[f_idx, 1])) ** 2

            return amp_loss + phi_loss

        n_opt = np.zeros(len(freqs), dtype=complex)
        for f_idx, freq in enumerate(freqs):
            if freq < 2.0:
                print(f"Frequency: {freq} (THz), (idx: {f_idx})")
                res = shgo(cost, bounds=bounds, args=(f_idx,), iters=7)
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

    n_simple = array([one, n_sub, n_opt, one], dtype=complex).T

    sam_tmm_simple_td, sam_tmm_simple_fd = calc_model(n_simple)

    if en_plot:
        plt.figure("Conductivity")
        plt.title("Conductivity " + plt_title)
        plt.plot(freqs[plot_range], sigma[plot_range].real, label="Real part")
        plt.plot(freqs[plot_range], sigma[plot_range].imag, label="Imaginary part")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Conductivity (S/m)")

        plt.figure("RI")
        plt.title(plt_title)
        plt.plot(freqs[plot_range], n_opt[plot_range].real, label="Refractive index (TMM)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Refractive index")

        plt.figure("Extinction coefficient")
        plt.title(plt_title)
        plt.plot(freqs[plot_range], n_opt[plot_range].imag, label="Extinction coefficient (TMM)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Extinction coefficient")

        noise_floor = get_noise_floor(s1_film_fd)

        plt.figure("Spectrum")
        plt.title(plt_title)
        plt.scatter(sam_tmm_simple_fd[plot_range1, 0],
                    to_db(sam_tmm_simple_fd[plot_range1, 1]) - noise_floor, label="Model (TMM)", zorder=2,
                    color="Green")

        phase_correction = lambda x: np.angle(x[:, 1])
        plt.figure("Phase")
        plt.title(plt_title)
        plt.plot(sam_tmm_simple_fd[plot_range, 0],
                 phase_correction(sam_tmm_simple_fd[plot_range, ]), label="Model (TMM)", linewidth=1.5)

        plt.figure("Time domain")
        plt.title(plt_title)
        plt.plot(sam_tmm_simple_td[:, 0], plot_td_scale * sam_tmm_simple_td[:, 1],
                    label=f"Model (TMM amplitude x{plot_td_scale})", color="Green", zorder=2, linewidth=2)

    def thin_film_model(n_sub, n_film, f_idx):
        omega_i = omega[f_idx]
        t01, t12, t23 = 2 / (1 + n_sub), 2 * n_sub / (n_sub + n_film), 2 * n_film / (n_sub + n_film)
        p1 = np.exp(-1j * omega_i * n_sub * d_sub / c_thz)
        p2 = np.exp(-1j * omega_i * n_film * d_film / c_thz)
        r10, r12, r23 = (n_sub - 1) / (n_sub + 1), (n_sub - n_film) / (n_sub + n_film), (n_film - 1) / (n_film + 1)

        enum = t01 * t12 * t23 * p1 * p2
        denum = (1 - p1 ** 2 * r12 * r10) * (1 + p2 ** 2 * r23 * r12)

        t = enum / denum

        return t

    def thin_film_model_fit():
        n_line = np.linspace(0, 800, 10000)

        def cost(p, f_idx):
            n_film = p[0] + 1j * p[1]
            t_func = thin_film_model(n_sub[f_idx], n_film, f_idx)  # * phase_shift[f_idx]
            # t_func_exp = s1_film_fd[:, 1] / s1_film_ref_fd[:, 1]

            amp_loss = (np.abs(t_func) - np.abs(s1_film_fd[f_idx, 1])) ** 2
            phi_loss = (np.angle(t_func) - np.angle(s1_film_fd[f_idx, 1])) ** 2

            return amp_loss + phi_loss

        n_opt = np.zeros(len(freqs), dtype=complex)
        for f_idx, freq in enumerate(freqs):
            if freq < 2.0:
                print(f"Frequency: {freq} (THz), (idx: {f_idx})")
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
            else:
                n_opt[f_idx] = n_opt[f_idx - 1]
                continue

        return n_opt

    """
    n_opt_tmm = n_opt
    
    n_opt = thin_film_model_fit()

    n_simple = array([one, n_sub, n_opt, one], dtype=complex).T

    sam_thin_film_td, sam_thin_film_fd = calc_model(n_simple)
    
    t_func = np.zeros_like(freqs, dtype=complex)
    for f_idx, freq in enumerate(freqs):
        t_func[f_idx] = thin_film_model(n_sub[f_idx], n_opt_tmm[f_idx], f_idx)

    sam_thin_film_fd = t_func * s1_film_ref_fd[:, 1]  # * phase_shift
    sam_thin_film_fd = array([freqs, sam_thin_film_fd]).T

    sam_thin_film_td = do_fft(sam_thin_film_fd)
    

    if en_plot:
        plt.figure("RI")
        plt.plot(freqs, n_opt.real, label="Thin film model")

        plt.figure("Extinction coefficient")
        plt.plot(freqs, n_opt.imag, label="Thin film model")

        plt.figure("Spectrum")
        plt.plot(sam_thin_film_fd[1:, 0], 20 * np.log10(np.abs(sam_thin_film_fd[1:, 1])), label="Thin film model")

        plt.figure("Phase")
        plt.plot(sam_thin_film_fd[1:, 0], phase_correction(sam_thin_film_fd[1:, ]), label="Thin film model")

        plt.figure("Time domain")
        plt.plot(sam_thin_film_td[:, 0], plot_td_scale*sam_thin_film_td[:, 1], label="Thin film model")
    """


if __name__ == '__main__':
    main()

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()

    plt.show()
