import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.constants import c as c0
from scipy.constants import epsilon_0
from functions_simple import do_fft, do_ifft, remove_offset, window, f_axis_idx_map, coh_tmm
from functions import to_db
from pathlib import Path
from scipy.optimize import shgo

from mpl_settings import mpl_style_params
mpl_style_params()

base_path_sub = Path(r"/home/ftpuser/ftp/Data/IPHT/Uncoated/s4")
base_path_film = Path(r"/home/ftpuser/ftp/Data/IPHT/s4_new_area/Image0")

sub_ref_file = base_path_sub / "2022-11-16T23-01-01.059419-5x5_sample_100avg-reference-X_6.000 mm-Y_10.000 mm.txt"
sub_file = base_path_sub / "2022-11-16T23-02-06.239541-5x5_sample_100avg-sample-X_40.000 mm-Y_10.000 mm.txt"
film_ref_file = base_path_film / "2023-08-05T04-52-46.699590-20avg-reference-X_95.000 mm-Y_-15.000 mm.txt"
film_file = base_path_film / "2023-08-05T04-53-16.599590-20avg-sample-X_10.000 mm-Y_-5.000 mm.txt"

sub_ref_td = np.loadtxt(sub_ref_file)
sub_td = np.loadtxt(sub_file)
film_ref_td = np.loadtxt(film_ref_file)
film_td = np.loadtxt(film_file)

# shift timeaxis to 0 ps
sub_ref_td[:, 0] -= sub_ref_td[0, 0]
sub_td[:, 0] -= sub_td[0, 0]
film_ref_td[:, 0] -= film_ref_td[0, 0]
film_td[:, 0] -= film_td[0, 0]

# subtract dc offset
sub_ref_td = remove_offset(sub_ref_td)
sub_td = remove_offset(sub_td)
film_ref_td = remove_offset(film_ref_td)
film_td = remove_offset(film_td)

# apply window
sub_ref_td = window(sub_ref_td, win_len=12, shift=0, en_plot=False, slope=0.99)
sub_td = window(sub_td, win_len=12, shift=0, en_plot=False, slope=0.99)
film_ref_td = window(film_ref_td, win_len=12, shift=0, en_plot=False, slope=0.99)
film_td = window(film_td, win_len=12, shift=0, en_plot=False, slope=0.99)

# fft
sub_ref_fd = do_fft(sub_ref_td, en_plots=False)
sub_fd = do_fft(sub_td, en_plots=False)
film_ref_fd = do_fft(film_ref_td, en_plots=False)
film_fd = do_fft(film_td, en_plots=False)

# calculate phase differences atan2(x) - atan2(y) = atan2(x/y)
phi_diff_sub = np.angle(sub_fd[:, 1] / sub_ref_fd[:, 1])
phi_diff_film = np.angle(film_fd[:, 1] / film_ref_fd[:, 1])

# constants
sub_shgo_bounds = [(1.6, 2.1), (0.01, 0.30)]  # [(n), (k)]

THz = 1e12
c_thz = c0 * 1e-6  # c in um / ps (1e6 / 1e12 = 1e-6) or um * THz
d_sub, d_film = 70, 0.2  # um
d_list_sub = [np.inf, d_sub, np.inf]  # um
d_list_film = [np.inf, d_sub, d_film, np.inf]  # um
angle_in = 0*pi/180  # rad

meas_plot_range = slice(10, 500)
f_axis = sub_ref_fd[:, 0].real  # THz
omega = 2 * pi * f_axis
# list of indices of the frequencies to be optimized
f_opt_idx = f_axis_idx_map(f_axis, (0.10, 3.0))

# In the sample measurement a part of the path (d_sam) is no longer air.
# So for phi(Er * t) to be equal to phi(Es) we have to subtract d_sam*n_air*omega/c from phi(Er).
# or equivalent from t, since phi(Er * t) = phi(Es) => phi(t) = phi(Er / Es)
phase_shift_sub = np.exp(-1j * 1 * d_sub * omega / c_thz)
phase_shift_film = np.exp(-1j * 1 * (d_sub + d_film) * omega / c_thz)


def optimize_sub(f_idx_, max_iters=8):
    def cost(p, ret_t=False):
        n = np.array([1, p[0] + 1j * p[1], 1], dtype=complex)

        lam_vac = c_thz / f_axis[f_idx_]
        t_tmm_fd_ = coh_tmm("s", n, d_list_sub, angle_in, lam_vac) * phase_shift_sub[f_idx_]

        sam_tmm_fd_ = t_tmm_fd_ * sub_ref_fd[f_idx_, 1]

        amp_loss = (np.abs(sam_tmm_fd_) - np.abs(sub_fd[f_idx_, 1])) ** 2
        phi_loss = (np.angle(t_tmm_fd_) - phi_diff_sub[f_idx_]) ** 2

        if ret_t:
            return t_tmm_fd_

        return amp_loss + phi_loss

    print(f"Frequency: {f_axis[f_idx_]} (THz), (idx: {f_idx_})")
    iters = 3
    res_ = shgo(cost, bounds=sub_shgo_bounds, iters=iters)
    while res_.fun > 1e-14:
        iters += 1
        res_ = shgo(cost, bounds=sub_shgo_bounds, iters=iters)
        if iters >= max_iters:
            break

    print(res_.x, res_.fun, "\n")
    t_tmm_fd = cost(res_.x, ret_t=True)

    return res_, t_tmm_fd


# substrate optimization loop
n_sub = np.ones_like(f_axis, dtype=complex)
t_tmm_sub = np.zeros_like(f_axis, dtype=complex)
for f_idx in f_opt_idx:
    f = f_axis[f_idx]

    if f < 0.15:
        opt_res, t_tmm_sub[f_idx] = optimize_sub(f_idx, max_iters=7)
        n_sub[f_idx] = opt_res.x[0] + 1j * opt_res.x[1]
    elif f <= 4.0:
        opt_res, t_tmm_sub[f_idx] = optimize_sub(f_idx, max_iters=7)
        n_sub[f_idx] = opt_res.x[0] + 1j * opt_res.x[1]
    else:
        n_sub[f_idx] = n_sub[f_idx - 1]


def optimize_film(f_idx_, bounds_, max_iters=8):
    def cost(p, ret_t=False):
        n = np.array([1, n_sub[f_idx_], p[0] + 1j * p[1], 1], dtype=complex)
        lam_vac = c_thz / f_axis[f_idx_]
        t_tmm_fd_ = coh_tmm("s", n, d_list_film, angle_in, lam_vac) * phase_shift_film[f_idx_]

        sam_tmm_fd = t_tmm_fd_ * film_ref_fd[f_idx_, 1]

        amp_loss = (np.abs(sam_tmm_fd) - np.abs(film_fd[f_idx_, 1])) ** 2
        phi_loss = (np.angle(t_tmm_fd_) - phi_diff_film[f_idx_]) ** 2

        if ret_t:
            return t_tmm_fd_

        return amp_loss + phi_loss

    print(f"Frequency: {f_axis[f_idx_]} (THz), (idx: {f_idx_})")
    iters = 7
    if f_axis[f_idx_] <= 0.150:
        res_ = shgo(cost, bounds=bounds_, iters=4)
    else:
        res_ = shgo(cost, bounds=bounds_, iters=5)
        while res_.fun > 1e-14:
            iters += 1
            res_ = shgo(cost, bounds=bounds_, iters=iters)
            if iters >= max_iters:
                break

    print(res_.x, res_.fun)
    t_tmm_fd = cost(res_.x, ret_t=True)

    return res_, t_tmm_fd


n_film = np.ones_like(f_axis, dtype=complex)
t_tmm_film = np.zeros_like(f_axis, dtype=complex)
for f_idx in f_opt_idx:
    f = f_axis[f_idx]

    if f <= 0.08:
        bounds = [(150, 350), (200, 320)]
        opt_res, t_tmm_film[f_idx] = optimize_film(f_idx, bounds)
        n_film[f_idx] = opt_res.x[0] + 1j * opt_res.x[1]
    elif (0.08 < f) * (f <= 0.5):
        bounds = [(60, 275), (20, 220)]
        opt_res, t_tmm_film[f_idx] = optimize_film(f_idx, bounds)
        n_film[f_idx] = opt_res.x[0] + 1j * opt_res.x[1]
    else:
        bounds = [(30, 100), (1, 80)]
        opt_res, t_tmm_film[f_idx] = optimize_film(f_idx, bounds)
        n_film[f_idx] = opt_res.x[0] + 1j * opt_res.x[1]

"""
n_film_ = n_film.copy()
n_film.real = n_film_.imag
n_film.imag = n_film_.real
"""

epsilon_r = n_film ** 2
sigma = 1j * (1 - epsilon_r) * epsilon_0 * omega * THz

phi_tmm_sub = np.angle(t_tmm_sub)
sub_tmm_fd = np.array([f_axis, t_tmm_sub * sub_ref_fd[:, 1]]).T
sub_tmm_td = do_ifft(sub_tmm_fd)

phi_tmm_film = np.angle(t_tmm_film)
film_tmm_fd = np.array([f_axis, t_tmm_film * film_ref_fd[:, 1]]).T
film_tmm_td = do_ifft(film_tmm_fd)

plt.figure("Time domain substrate")
plt.title("Time domain substrate")
plt.plot(sub_tmm_td[:, 0], sub_tmm_td[:, 1], label="TMM", linewidth=2)
plt.plot(sub_ref_td[:, 0], sub_ref_td[:, 1], label="Ref Meas", linewidth=2)
plt.plot(sub_td[:, 0], sub_td[:, 1], label="Sam Meas", linewidth=2)
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude (arb. u.)")

plt.figure("Spectrum substrate")
plt.title("Spectrum substrate")
plt.plot(sub_ref_fd[meas_plot_range, 0], to_db(sub_ref_fd[meas_plot_range, 1]), label="Reference")
plt.plot(sub_fd[meas_plot_range, 0], to_db(sub_fd[meas_plot_range, 1]), label="Substrate")
plt.plot(sub_tmm_fd[f_opt_idx, 0], to_db(sub_tmm_fd[f_opt_idx, 1]), label="TMM fit", zorder=2)
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")

plt.figure("Phase substrate")
plt.title("Phases substrates")
plt.plot(f_axis[meas_plot_range], phi_diff_sub[meas_plot_range], label="Measured", linewidth=1.5)
plt.plot(f_axis[f_opt_idx], phi_tmm_sub[f_opt_idx], label="TMM", linewidth=1.5)
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")

fig = plt.figure("Refractive index substrate")
ax_ = fig.add_subplot(111)
ax_.set_ylabel("Refractive index substrate")
ax_.set_xlabel("Frequency (THz)")
ax_.plot(f_axis[f_opt_idx], n_sub[f_opt_idx].real, label="Re(n)")
ax_.plot(f_axis[f_opt_idx], n_sub[f_opt_idx].imag, label="Im(n)")

plt.figure("Time domain film")
plt.title("Time domain film")
plt.plot(film_tmm_td[:, 0], film_tmm_td[:, 1], label="TMM", linewidth=2)
plt.plot(film_ref_td[:, 0], film_ref_td[:, 1], label="Ref Meas", linewidth=2)
plt.plot(film_td[:, 0], film_td[:, 1], label="Sam Meas", linewidth=2)
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude (arb. u.)")

plt.figure("Spectrum film")
plt.title("Spectrum film")
plt.plot(film_ref_fd[meas_plot_range, 0], to_db(film_ref_fd[meas_plot_range, 1]), label="Reference")
plt.plot(film_tmm_fd[f_opt_idx, 0], to_db(film_tmm_fd[f_opt_idx, 1]), label="TMM fit", zorder=2)
plt.plot(film_fd[meas_plot_range, 0], to_db(film_fd[meas_plot_range, 1]), label="Sam meas")
plt.xlabel("Frequency (THz)")
plt.ylabel("Amplitude (dB)")

plt.figure("Phase film")
plt.title("Phases film")
plt.plot(f_axis[meas_plot_range], phi_diff_film[meas_plot_range], label="Measured", linewidth=1.5)
plt.plot(f_axis[f_opt_idx], phi_tmm_film[f_opt_idx], label="TMM", linewidth=1.5)
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase (rad)")

fig = plt.figure("Refractive index film")
ax_ = fig.add_subplot(111)
ax_.set_ylabel("Refractive index film")
ax_.set_xlabel("Frequency (THz)")
ax_.plot(f_axis[f_opt_idx], n_film[f_opt_idx].real, label="Re(n)")
ax_.plot(f_axis[f_opt_idx], n_film[f_opt_idx].imag, label="Im(n)")

fig = plt.figure("cond_spectrum")
ax_ = fig.add_subplot(111)
ax_.set_ylabel("Conductivity (S/m)")
ax_.set_xlabel("Frequency (THz)")
ax_.plot(f_axis[f_opt_idx], sigma[f_opt_idx].real, label="Re($\sigma$)")
ax_.plot(f_axis[f_opt_idx], sigma[f_opt_idx].imag, label="Im($\sigma$)")

for fig_label in plt.get_figlabels():
    plt.figure(fig_label)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        plt.legend()

plt.show()
