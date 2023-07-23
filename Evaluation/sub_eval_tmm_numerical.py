import random
from load_teralyzervalues import teralyzer_read_point
import numpy as np
from scipy.optimize import shgo
from consts import *
from numpy import array
import matplotlib.pyplot as plt
from tmm_slim import coh_tmm
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

    sub_ref_fd = phase_correction(sub_ref_fd, fit_range=(0.60, 1.60), extrapolate=True, ret_fd=True)
    sub_fd = phase_correction(sub_fd, fit_range=(0.60, 1.60), extrapolate=True, ret_fd=True)

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
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)
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
                t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)
                sam_tmm_fd = t_tmm_fd * sub_ref_fd[f_idx, 1] * phase_shift[f_idx]

                amp_loss = (np.abs(sam_tmm_fd) - np.abs(sub_fd[f_idx, 1])) ** 2
                phi_loss = (np.angle(sam_tmm_fd) - np.angle(sub_fd[f_idx, 1])) ** 2

                return amp_loss + phi_loss

            def optimize(f_idx_, max_iters=8):
                iters = shgo_iters - 3
                print(f"Frequency: {freqs[f_idx_]} (THz), (idx: {f_idx_})")
                res = shgo(cost, bounds=bounds, args=(f_idx_,), iters=iters - 2)
                while res.fun > 1e-10:
                    iters += 1
                    res = shgo(cost, bounds=bounds, args=(f_idx_,), iters=iters)
                    if iters >= max_iters:
                        break

                print(res.x, res.fun)

                return res

            if single_f_idx is not None:
                res = optimize(single_f_idx)

                return res.x[0] + 1j * res.x[1]
            else:
                n_sub = np.zeros(len(freqs), dtype=complex)
                for f_idx, freq in enumerate(freqs):
                    if freq < 0.15:
                        res = optimize(f_idx, max_iters=5)

                        n_sub[f_idx] = res.x[0] + 1j * res.x[1]
                    elif freq <= 4:
                        res = optimize(f_idx)

                        n_sub[f_idx] = res.x[0] + 1j * res.x[1]
                    else:
                        n_sub[f_idx] = n_sub[f_idx-1]

        n_shgo = array([one, n_sub, one]).T

        sam_tmm_shgo_td, sam_tmm_shgo_fd = calc_model(n_shgo)

        phi_tmm = phase_correction(sam_tmm_shgo_fd, disable=True, fit_range=(0.55, 1.0))

        absorption = 2*n_sub.imag*omega*THz/c0

        n_tera = teralyzer_read_point(*eval_point)

        label = f"{sub_image.name} (TMM) x={eval_point[0]} mm, y={eval_point[1]} mm"
        if en_plot:
            plt.figure("RI")
            plt.title("Complex refractive index substrates")
            plt.plot(freqs[plot_range_sub], n_sub[plot_range_sub].real, label="Real part " + label)
            plt.plot(freqs[plot_range_sub], n_sub[plot_range_sub].imag, label="Imaginary part " + label)
            #plt.plot(n_tera[:, 0], n_tera[:, 1].real, label=f"Real part (Teralyzer)\nx={eval_point[0]} mm, y={eval_point[1]} mm")
            #plt.plot(n_tera[:, 0], n_tera[:, 1].imag, label=f"Imaginary part (Teralyzer)\nx={eval_point[0]} mm, y={eval_point[1]} mm")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Refractive index")

            plt.figure("Extinction coefficient")
            plt.title("Extinction coefficient substrates")
            plt.plot(freqs[plot_range_sub], n_sub[plot_range_sub].imag, label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Extinction coefficient")

            plt.figure("Absorption")
            plt.title("Absorption substrates")
            plt.plot(freqs[plot_range_sub], 0.01*absorption[plot_range_sub], label=label)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Absorption (1/cm)")

            noise_floor = np.mean(20 * np.log10(np.abs(sub_ref_fd[sub_ref_fd[:, 0] > 6.0, 1])))

            plt.figure("Spectrum")
            plt.title("Spectrum substrate")
            plt.plot(sub_ref_fd[plot_range1, 0], to_db(sub_ref_fd[plot_range1, 1]) - noise_floor, label="Reference")
            plt.plot(sam_tmm_shgo_fd[plot_range1, 0], to_db(sam_tmm_shgo_fd[plot_range1, 1])-noise_floor, label="Uncoated substrate")
            plt.plot(sub_fd[plot_range1, 0], to_db(sub_fd[plot_range1, 1])-noise_floor, label="TMM fit", zorder=2)
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Phase")
            plt.title("Phases substrates")
            plt.plot(sam_tmm_shgo_fd[plot_range_sub, 0], phi_tmm[plot_range_sub, 1],
                     label=label, linewidth=1.5)

            plt.figure("Time domain")
            plt.plot(sam_tmm_shgo_td[:, 0], sam_tmm_shgo_td[:, 1], label=label, linewidth=2)

            fig, ax1 = plt.subplots()
            ax1.set_title("Uncoated substrate")
            color = 'tab:red'
            ax1.set_xlabel('Frequency (THz)')
            ax1.set_ylabel('Refractive index', color=color)
            ax1.plot(freqs[plot_range_sub], n_sub[plot_range_sub].real, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            #ax1.grid(color='r')

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('Absorption (1/cm)', color=color)  # we already handled the x-label with ax1
            ax2.plot(freqs[plot_range_sub], 0.01*absorption[plot_range_sub], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            #ax2.grid(color='blue')

            fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return n_sub


if __name__ == '__main__':
    from Measurements.image import Image

    sample_idx = 3
    # image_data = data_dir / "Edge" / sample_names[sample_idx]
    image_data = data_dir / "Coated" / sample_names[sample_idx]
    image = Image(image_data)
    image.plot_point(33, 11)
    #image.plot_image(quantity="p2p", img_extent=[-10, 30, 0, 20])
    image.plot_image(quantity="p2p")

    """
    for i in range(4):
        sample_idx = 3
        image_data = data_dir / "Uncoated" / sample_names[sample_idx]
        image = Image(image_data)
        
        while True:
            eval_point = random.choice(image.all_points)
            if (20 < eval_point[0]) * (eval_point[0] < 50):
                if (0 < eval_point[1]) * (eval_point[1] < 20):
                    break

        n_sub = tmm_eval(sub_image=image, eval_point=eval_point, en_plot=True)
        #image.plot_point(*eval_point, label=f"Sample {sam_idx + 1} Uncoated")
    """

    # eval_point = (20, 10)  # used for s1-s3
    # eval_point = (33, 11)  # s4
    # eval_point = (42, 20)
    eval_point = (54, 17)

    # n_sub = tmm_eval(sub_image=image, eval_point=eval_point, en_plot=True)

    # np.save(f"n_sub_s{sam_idx + 1}_{eval_point[0]}_{eval_point[1]}.npy", n_sub)

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
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            plt.legend()

    plt.show()
