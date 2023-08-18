from tmm import coh_tmm
from consts import *
from image import Image
from mpl_settings import *
import matplotlib.pyplot as plt
from functions import unwrap, do_fft, do_ifft, to_db, get_noise_floor

# should the unwrapped phase also be linear for multilayer samples ?
# -> take ref and send it through 2 layer sample -> unwrap resulting phase
# result: lgtm


def main():
    sample_idx = 0

    meas_dir_sub = data_dir / "Uncoated" / sample_names[sample_idx]
    sub_image = Image(data_path=meas_dir_sub)

    ref_td, ref_fd = sub_image.get_ref(both=True)

    sub_image.plot_point(x=10, y=10, sub_noise_floor=True)

    freqs = ref_fd[:, 0].real

    omega = 2*pi*freqs
    phase_shift = np.exp(-1j * d_sub * omega / c_thz)

    one = np.ones_like(freqs)
    n_sub = (1.6+1j*0.01) * one
    n_film = (200 + 1j*200) * one

    n_list = array([one, n_sub, n_film, one], dtype=complex).T
    d_list = [inf, d_sub, film_thicknesses[0], inf]

    ts_tmm_fd = np.zeros_like(freqs, dtype=complex)
    for f_idx, freq in enumerate(freqs):
        lam_vac = c_thz / freq
        n = n_list[f_idx]
        t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
        ts_tmm_fd[f_idx] = t_tmm_fd

    sam_tmm_fd = array([freqs, ts_tmm_fd * ref_fd[:, 1] * phase_shift]).T
    sam_tmm_td = do_ifft(sam_tmm_fd)

    phi_tmm = unwrap(sam_tmm_fd)
    fit_slice = (0.25 < freqs) * (freqs < 4.5)
    res = np.polyfit(freqs[fit_slice], phi_tmm[fit_slice, 1], 1, full=True)
    print(100*(phi_tmm[450, 1] - phi_tmm[25, 1])/425)
    print(res)

    plt.figure("Spectrum")
    plt.title("Spectrum substrates")
    plt.plot(sam_tmm_fd[plot_range1, 0], to_db(sam_tmm_fd[plot_range1, 1]), label="tmm", zorder=2,
             color="Green")

    plt.figure("Phase")
    plt.title("Phases substrates")
    plt.plot(sam_tmm_fd[plot_range_sub, 0], phi_tmm[plot_range_sub, 1],
             label="tmm", linewidth=1.5)

    plt.figure("Time domain")
    plt.plot(sam_tmm_td[:, 0], sam_tmm_td[:, 1], label="tmm", linewidth=2)


if __name__ == '__main__':
    main()

    for fig_label in plt.get_figlabels():
        if "Sample" in fig_label:
            continue
        plt.figure(fig_label)
        plt.legend()

    plt.show()

