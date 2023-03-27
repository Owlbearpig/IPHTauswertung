import numpy as np
from scipy.optimize import shgo
from consts import *
from numpy import array
from matplotlib import ticker
from Measurements.measurements import get_all_measurements
from Measurements.image import Image
import matplotlib.pyplot as plt
from tmm_slim import coh_tmm
from functions import do_ifft, filtering, do_fft, phase_correction, to_db, get_noise_floor, window, unwrap, zero_pad
from sub_eval_tmm_numerical import tmm_eval
from mpl_settings import *


def drude_fit(sigma_exp, omega, sample_idx):
    f0, f1 = drude_fit_range
    freqs = omega / (2 * pi)
    freq_slice = (f0 < freqs) * (freqs < f1)

    freqs = freqs[freq_slice]
    omega = omega[freq_slice]
    sigma_exp = sigma_exp[freq_slice]

    def drude_model(dc, tau, omega):
        return dc / (1 - 1j * omega * tau)

    def cost(p):
        dc, tau = p
        sigma_model = drude_model(dc, tau, omega)

        loss = np.sum((sigma_model.real - sigma_exp.real) ** 2 + (sigma_model.imag - sigma_exp.imag) ** 2)

        loss = np.sum((sigma_model.real - sigma_exp.real) ** 2)
        # loss = np.sum((sigma_model.imag - sigma_exp.imag) ** 2)

        return loss

    bounds = shgo_bounds_drude[sample_idx]

    res = shgo(cost, bounds=bounds, iters=6)
    print(res)

    sigma_model = drude_model(*res.x, omega)

    plt.figure("Drude fit")
    plt.plot(freqs, sigma_exp.real, label="Measurement (real)")
    plt.plot(freqs, sigma_exp.imag, label="Measurement (imaginary)")
    plt.plot(freqs, sigma_model.real, label="Drude (real)")
    plt.plot(freqs, sigma_model.imag, label="Drude (imaginary)")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Conductivity (S/m)")

    dc, tau = res.x

    return dc, tau




# -2, 16
def main(en_plot=True, sample_idx=0, eval_point=None, d_film=None):
    if d_film is None:
        d_film = sample_thicknesses[sample_idx]

    plot_td_scale = td_scales[sample_idx]

    if eval_point is None:
        eval_point = (30.0, 10.0)

    plt_title = f"Sample {sample_idx + 1} {sample_labels[sample_idx]} (x={eval_point[0]} mm, y={eval_point[1]} mm)"

    data_dir_film = data_dir / "Coated" / sample_names[sample_idx]

    image = Image(data_dir_film)
    film_td = image.get_point(*eval_point, sub_offset=True, add_plot=False)
    film_ref_td = image.get_ref(both=False, coords=eval_point)

    film_td = window(film_td, win_len=12, shift=0, en_plot=False, slope=0.05)
    film_ref_td = window(film_ref_td, win_len=12, shift=0, en_plot=False, slope=0.05)

    film_ref_fd, film_fd = do_fft(film_ref_td), do_fft(film_td)

    film_ref_td, film_ref_fd = phase_correction(film_ref_fd, fit_range=(0.8, 1.6), extrapolate=True,
                                   en_plot=False, both=True)
    film_td, film_fd = phase_correction(film_fd, fit_range=(0.8, 1.6), extrapolate=True,
                               en_plot=False, both=True)

    image.plot_point(*eval_point, sam_td=film_td, ref_td=film_ref_td,
                     label=f"Sample {sample_idx + 1}", td_scale=plot_td_scale, sub_noise_floor=True)

    data_dir_film = data_dir / "Uncoated" / sample_names[sample_idx]
    image_sub = Image(data_dir_film)

    #"""

    one2onesub = True
    if one2onesub:
        try:
            n_sub = np.load(f"n_sub_s{sample_idx}_{eval_point[0]}_{eval_point[1]}.npy")
        except FileNotFoundError:
            n_sub = tmm_eval(image_sub, eval_point=eval_point)
            np.save(f"n_sub_s{sample_idx}_{eval_point[0]}_{eval_point[1]}.npy", n_sub)
    else:
        try:
            n_sub = np.load(f"n_sub_s{sample_idx + 1}_9_9.npy")
        except FileNotFoundError:
            n_sub = tmm_eval(image_sub, eval_point=eval_point)
            np.save(f"n_sub_s{sample_idx + 1}_9_9.npy", n_sub)

    # n_sub *= np.random.random(n_sub.shape)

    freqs = film_ref_fd[:, 0].real
    one = np.ones_like(freqs)
    d_list = [inf, d_sub, d_film, inf]

    omega = 2 * pi * freqs
    phase_shift = np.exp(-1j * (d_sub + d_film) * omega / c_thz)

    measurement = image.get_measurement(*eval_point)
    ref_interpol_fd = np.zeros_like(freqs, dtype=complex)
    for f_idx, freq in enumerate(freqs):
        ref_interpol_fd[f_idx] = image._ref_interpolation(measurement, selected_freq_=freqs[f_idx], ret_cart=True)

    def calc_model(n_list):
        ts_tmm_fd = np.zeros_like(freqs, dtype=complex)
        for f_idx, freq in enumerate(freqs):
            lam_vac = c_thz / freq
            n = n_list[f_idx]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)
            ts_tmm_fd[f_idx] = t_tmm_fd

        sam_tmm_fd = array([freqs, ts_tmm_fd * film_ref_fd[:, 1] * phase_shift]).T
        #sam_tmm_fd = array([freqs, ts_tmm_fd * ref_interpol_fd * phase_shift]).T
        sam_tmm_td = do_ifft(sam_tmm_fd)

        return sam_tmm_td, sam_tmm_fd

    def simple_fit():
        bounds = shgo_bounds_film[sample_idx]

        def cost(p, f_idx):
            n = array([1, n_sub[f_idx], p[0] + 1j * p[1], 1])
            lam_vac = c_thz / freqs[f_idx]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)

            # sam_tmm_fd = t_tmm_fd * film_ref_interpol_fd * phase_shift[f_idx]
            sam_tmm_fd = t_tmm_fd * film_ref_fd[f_idx, 1] * phase_shift[f_idx]

            amp_loss = (np.abs(sam_tmm_fd) - np.abs(film_fd[f_idx, 1])) ** 2
            phi_loss = (np.angle(sam_tmm_fd) - np.angle(film_fd[f_idx, 1])) ** 2

            return amp_loss + phi_loss

        n_film = np.zeros(len(freqs), dtype=complex)
        for f_idx, freq in enumerate(freqs):
            if freq <= 2.0:
                print(f"Frequency: {freq} (THz), (idx: {f_idx})")
                if freq <= 0.25:
                    res = shgo(cost, bounds=bounds, args=(f_idx,), iters=4)
                else:
                    iters = shgo_iters - 3
                    res = shgo(cost, bounds=bounds, args=(f_idx,), iters=iters - 2)
                    while res.fun > 1e-8:
                        iters += 1
                        res = shgo(cost, bounds=bounds, args=(f_idx,), iters=iters)
                        if iters >= 8:
                            break

                n_film[f_idx] = res.x[0] + 1j * res.x[1]
                print(n_film[f_idx], f"Fun: {res.fun}", "\n")
            else:
                n_film[f_idx] = n_film[f_idx - 1]

        return n_film

    x, y = eval_point
    try:
        n_film = np.load(f"n_opt_simple_fit_s{sample_idx + 1}_{x}_{y}.npy")
        raise FileNotFoundError
        #n_film_300 = np.load(f"n_opt_simple_fit_s{sample_idx + 1}_{x}_{y}_300nm.npy")
        #n_film_200 = np.load(f"n_opt_simple_fit_s{sample_idx + 1}_{x}_{y}_200nm.npy")
    except FileNotFoundError:
        n_film = simple_fit()
        np.save(f"n_opt_simple_fit_s{sample_idx + 1}_{x}_{y}.npy", n_film)
        #n_film_300 = simple_fit()
        #np.save(f"n_opt_simple_fit_s{sample_idx + 1}_{x}_{y}_300nm.npy", n_film_300)

    epsilon_film = n_film ** 2

    #epsilon_200 = n_film_200 ** 2
    #epsilon_300 = n_film_300 ** 2
    #sigma_200 = 1j * (1 - epsilon_200) * epsilon_0 * omega * THz
    #sigma_300 = 1j * (1 - epsilon_300) * epsilon_0 * omega * THz
    # sigma_dc, tau = drude_fit(sigma_300, omega, sample_idx)

    #n_simple_300 = array([one, n_sub, n_film_300, one], dtype=complex).T
    #n_simple_200 = array([one, n_sub, n_film_200, one], dtype=complex).T

    sigma = 1j * (1 - epsilon_film) * epsilon_0 * omega * THz
    n_simple = array([one, n_sub, n_film, one], dtype=complex).T
    sam_tmm_simple_td, sam_tmm_simple_fd = calc_model(n_simple)

    #sam_tmm_simple_td_300, sam_tmm_simple_fd_300 = calc_model(n_simple_300)
    #sam_tmm_simple_td_200, sam_tmm_simple_fd_200 = calc_model(n_simple_200)

    def fmt(x, val):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    if en_plot:
        plt.figure("Conductivity old")
        plt.title("Conductivity " + plt_title)
        plt.ticklabel_format(scilimits=(-2, 3))
        plt.plot(freqs[plot_range], sigma[plot_range].real, label="Real part")
        plt.plot(freqs[plot_range], sigma[plot_range].imag, label="Imaginary part")
        #plt.plot(freqs[plot_range], sigma_200[plot_range].real, label="Real part $d_{film}=$ 200 nm")
        #plt.plot(freqs[plot_range], sigma_300[plot_range].real, label="Real part $d_{film}=$ 300 nm")
        #plt.plot(freqs[plot_range], sigma_200[plot_range].imag, ":", label="Imaginary part $d_{film}=$ 200 nm")
        #plt.plot(freqs[plot_range], sigma_300[plot_range].imag, ":", label="Imaginary part $d_{film}=$ 300 nm")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Conductivity (S/m)")

        plt.figure("RI")
        plt.title(plt_title)
        plt.plot(freqs[plot_range], n_film[plot_range].real, label="Refractive index (TMM)")
        plt.plot(freqs[plot_range], n_film[plot_range].imag, label="Extinction coefficient (TMM)")
        #plt.plot(freqs[plot_range], n_film_200[plot_range].real, label="Refractive index $d_{film}=$ 200 nm(TMM)")
        #plt.plot(freqs[plot_range], n_film_300[plot_range].real, label="Refractive index $d_{film}=$ 300 nm(TMM)")
        #plt.plot(freqs[plot_range], n_film_200[plot_range].imag, ":", label="Extinction coefficient $d_{film}=$ 200 nm(TMM)")
        #plt.plot(freqs[plot_range], n_film_300[plot_range].imag, ":", label="Extinction coefficient $d_{film}=$ 300 nm(TMM)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Complex refractive index")

        plt.figure("Extinction coefficient")
        plt.title(plt_title)
        plt.plot(freqs[plot_range], n_film[plot_range].imag, label="Extinction coefficient (TMM)")
        #plt.plot(freqs[plot_range], n_film[plot_range].imag, label="Extinction coefficient $d_{film}=$ 200 nm(TMM)")
        #plt.plot(freqs[plot_range], n_film[plot_range].imag, label="Extinction coefficient $d_{film}=$ 300 nm(TMM)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Extinction coefficient")

        noise_floor = get_noise_floor(film_fd)

        plt.figure("Spectrum")
        plt.title(plt_title)
        plt.scatter(sam_tmm_simple_fd[plot_range, 0], to_db(sam_tmm_simple_fd[plot_range, 1]) - noise_floor,
                    label="Model (TMM)", zorder=2, color="Green")
        #plt.scatter(sam_tmm_simple_fd_200[plot_range, 0], to_db(sam_tmm_simple_fd_200[plot_range, 1]) - noise_floor, label="Model (TMM)", zorder=2, color="Green")

        plt.figure("Phase")
        plt.title(plt_title)
        plt.plot(sam_tmm_simple_fd[plot_range, 0], unwrap(sam_tmm_simple_fd)[plot_range, 1],
                 label="Model (TMM)", linewidth=1.5)

        plt.figure("Time domain")
        plt.title(plt_title)
        plt.plot(sam_tmm_simple_td[:, 0], plot_td_scale * sam_tmm_simple_td[:, 1],
                 label=f"Model (TMM amplitude x{plot_td_scale})", color="Green", zorder=2, linewidth=2)

    sigma_ = np.array([freqs, sigma], dtype=complex).T

    return sigma_


if __name__ == '__main__':
    # main(sample_idx=3, eval_point=(24, 23))
    #main(sample_idx=0, eval_point=(24, 23))
    sample_idx = 1
    d_film = sample_thicknesses[sample_idx]
    d_film *= 1.0
    eval_point = (24.0, 23.0)

    sigma0 = main(sample_idx=sample_idx, eval_point=eval_point, d_film=d_film)
    sigma_plus = main(sample_idx=sample_idx, eval_point=eval_point, d_film=d_film*0.9)
    sigma_minus = main(sample_idx=sample_idx, eval_point=eval_point, d_film=d_film*1.1)

    sigma0_fft = np.fft.fft(sigma0[plot_range].real)
    plt.figure("fft sigma")
    plt.title(f"{d_film*1e6} nm")
    plt.plot(np.abs(sigma0_fft))

    freqs = sigma0[:, 0].real
    plt_title = f"Sample {sample_idx + 1} {sample_labels[sample_idx]}\n(x={eval_point[0]} mm, y={eval_point[1]} mm)"

    plt.figure("Conductivity")
    plt.title("Conductivity " + plt_title)# + f"{d_film*1e6} nm")
    plt.ticklabel_format(scilimits=(-2, 3))
    plt.fill_between(freqs[plot_range], sigma0[plot_range, 1].real, sigma_minus[plot_range, 1].real, alpha=0.5, color="blue", label=r"$d_{film}$ $\pm 10$ %")
    plt.plot(freqs[plot_range], sigma0[plot_range, 1].real, label="Real part", color="blue")
    plt.fill_between(freqs[plot_range], sigma0[plot_range, 1].real, sigma_plus[plot_range, 1].real, alpha=0.5, color="blue")

    plt.fill_between(freqs[plot_range], sigma0[plot_range, 1].imag, sigma_minus[plot_range, 1].imag, alpha=0.5, color="red", label=r"$d_{film}$ $\pm 10$ %")
    plt.plot(freqs[plot_range], sigma0[plot_range, 1].imag, label="Imaginary part", color="red")
    plt.fill_between(freqs[plot_range], sigma0[plot_range, 1].imag, sigma_plus[plot_range, 1].imag, alpha=0.5, color="red")

    plt.xlabel("Frequency (THz)")
    plt.ylabel("Conductivity (S/m)")

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()

    plt.show()
