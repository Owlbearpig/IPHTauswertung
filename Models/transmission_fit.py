import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, array
from matplotlib.widgets import Slider, Button
from scipy.constants import elementary_charge, electron_mass, epsilon_0
from Measurements.image import Image
from consts import data_dir, c_thz, angle_in, shgo_bounds_film
from mpl_settings import mpl_style_params
from tmm_slim import coh_tmm
from tmm import inc_tmm
from sub_eval_tmm_numerical import tmm_eval
from functions import f_axis_idx_map, save_array_as_csv, window

mpl_style_params()

sample_idx = 0  # 0, 3

if sample_idx == 0:
    # film_eval_pt = (5, -2) # original # s1 or s4??
    film_eval_pt = (7, 7)  # original (7, 7)
    # film_eval_pt = (7.5, 8.5)  # (7.5, 8.5) works well? 6, 84 kS/cm
    save_file = "signal_Ag"
elif sample_idx == 3:
    film_eval_pt = (8, -14)
    exp_file_name = ""
    save_file = "signal_ITO"
else:
    exit()

sub_eval_pt = (30, 10)  # (37.5, 18.5) # high p2p
sub_eval_pt = (37.5, 18.5)
freq_range_ = (0.15, 3.0)

# meas_dir_sub = data_dir / "Uncoated" / f"s{sample_idx+1}"
meas_dir_sub = data_dir / "Uncoated" / f"s{sample_idx + 1}"
sub_image = Image(data_path=meas_dir_sub, options={"load_mpl_style": False}, sample_idx=sample_idx)
# sub_image.plot_image()

n_sub = tmm_eval(sub_image, eval_point_=sub_eval_pt, en_plot=False, freq_range=freq_range_)
# n_sub[:, 1].imag = 0.09  # 0 with scattering on ??
# n_sub[:, 1].imag = 0.45
# n_sub[:, 1].real = 1.68

# plt.figure("aaa")
# plt.plot(n_sub[:, 1].real)

if sample_idx == 3:
    meas_dir_film = data_dir / "s4_new_area" / "Image0"
elif sample_idx == 0:
    meas_dir_film = data_dir / "s1_new_area" / "Image3_28_07_2023"
else:
    exit("_")

options = {"cbar_min": 1e5, "cbar_max": 3.5e5, "log_scale": False, "color_map": "viridis",
           "load_mpl_style": False, "invert_x": True, "invert_y": True}  # s4
"""
options = {"cbar_min": 5e5, "cbar_max": 1.5e7, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": True}  # s1
"""
film_image = Image(meas_dir_film, sub_image, sample_idx, options)
film_image.plot_point(*film_eval_pt, save_file=save_file, sub_noise_floor=True)

ref_td, ref_fd = film_image.get_ref(coords=film_eval_pt, both=True)
film_td, film_fd = film_image.get_point(*film_eval_pt, both=True)

save_file = f"t_abs_x{film_eval_pt[0]}_y{film_eval_pt[1]}_s{sample_idx + 1}_.npy"

try:
    t_abs_meas = np.load(save_file)
except FileNotFoundError:
    t_abs, t_abs_meas = film_image.plot_transmittance(*film_eval_pt, freq_range=freq_range_)
    np.save(save_file, t_abs_meas)
"""
t_abs_meas_film = film_image.get_transmittance(*film_eval_pt, freq_range=freq_range_)
t_abs_meas_sub = sub_image.get_transmittance(*sub_eval_pt, freq_range=freq_range_)
plt.figure("add")
plt.plot(20*np.log10(t_abs_meas_sub[:, 1]))
plt.plot(0.10*20*np.log10(t_abs_meas_film[:, 1]))
"""
freq_axis = n_sub[:, 0].real
freq_axis_idx = f_axis_idx_map(freq_axis, freq_range_)

freq_axis = freq_axis[freq_axis_idx]
omega = 2 * pi * freq_axis

# init_sigma0 = 3.38  # 1 / (mOhm cm)
# init_sigma0 = 3.12
if sample_idx == 3:
    init_tau_d_ = 5
    init_sigma0 = 1.07  # 2.60
    # init_sigma0 = 160  # 1 / (mOhm cm)
    # init_tau = 0.0022  # mm
    # init_tau = 2.2  # um
    init_tau = 12.4  # 4.5  # um
elif sample_idx == 0:
    init_tau_d_ = 5
    init_sigma0 = 116.4  # 81.7
    # init_sigma0 = 160  # 1 / (mOhm cm)
    # init_tau = 0.0131  # mm
    # init_tau = 28  # um
    init_tau = 11.8  # 6.8  # um
else:
    exit("00")


def transmission_simple(freq_axis_, sigma0_, tau_, **kwargs):
    tau_ *= 1e-3  # um -> mm. 1 um = 1e-3 mm
    sigma0_ = 1e5 * sigma0_  # 1/(mOhm cm) (1/(1e-3*1e-2)) = 1e5 -> S / m
    d_list = array([np.inf, 0.070, 0.0002, np.inf], dtype=float)  # in mm
    # d_list = array([np.inf, 0.430, 0.000088, np.inf], dtype=float)  # in mm
    n_film_ = (1 + 1j) * np.sqrt(sigma0_ / (2 * epsilon_0 * omega * 1e12))
    # n_film_ = np.ones_like(omega) * (35+1j*35)
    n_sub_ = n_sub[freq_axis_idx, 1]
    # n_sub_ = np.ones_like(freq_axis_) * 3.4175

    plt.figure("Film refractive index")
    plt.title("Film refractive index")
    plt.plot(freq_axis_, n_film_.real, label="RIdx real")
    plt.plot(freq_axis_, n_film_.imag, label="RIdx imag")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")

    t12 = 2 * 1 / (n_film_ + 1)
    r12 = (1 - n_film_) / (n_film_ + 1)

    t23 = 2 * n_film_ / (n_sub_ + n_film_)
    r23 = (n_film_ - n_sub_) / (n_sub_ + n_film_)

    t34 = 2 * n_sub_ / (1 + n_sub_)
    r34 = (n_sub_ - 1) / (1 + n_sub_)

    """
    plt.figure("t")
    plt.plot(np.abs(t12), label="t12")
    plt.plot(np.abs(t23), label="t23")
    plt.plot(np.abs(t34), label="t34")

    plt.figure("r")
    plt.plot(np.abs(r12), label="r12")
    plt.plot(np.abs(r23), label="r23")
    plt.plot(np.abs(r34), label="r34")

    plt.figure("exp")
    plt.plot(np.abs(np.exp(1j * n_film_ * omega * d_list[2] / c_thz)), label="1")
    plt.plot(np.abs(np.exp(1j * n_sub_ * omega * d_list[1] / c_thz)), label="2")
    plt.plot(np.abs(np.exp(2 * 1j * omega * d_list[2] * n_film_ / c_thz)), label="3")
    """
    lam_vac = c_thz / freq_axis_
    # alph_scat = (1 / d_list[1]) * ((n_sub_ - 1) * 4 * pi * tau_ / lam_vac**2)
    # alph_scat = (4 * pi * tau_ / lam_vac**2) * (n_sub_ - 1)
    alph_scat = ((4 * pi * tau_ / lam_vac) * (n_sub_ - 1))**2  # "Ralf scattering"
    # alph_scat = (2 * pi**2 * tau_**2 / lam_vac ** (3/2)) * (n_sub_**2 - 1) / (n_sub_**2 + 1)
    # alph_scat = (1 * 4 * pi * tau_ / lam_vac ** 2) * (n_sub_ ** 2 - 1) / (n_sub_ ** 2 + 2)
    ampl_att_ = np.abs(np.exp(-alph_scat))

    # r34 *= ampl_att_
    r23 *= ampl_att_
    t23 *= ampl_att_
    r12 *= ampl_att_

    phi_s = 1j * n_sub_ * omega * d_list[1] / c_thz
    phi_f = 1j * n_film_ * omega * d_list[2] / c_thz

    t_enu = t12 * t23 * t34 * np.exp(phi_f) * np.exp(phi_s)

    # t_den = 1 + r12 * r23 * np.exp(2 * 1j * omega * d_list[2] * n_film_ / c_thz)
    t_den = 1 + r12 * r23 * np.exp(2 * phi_f)
    t_den += r34 * r23 * np.exp(2 * phi_s)
    t_den += r12 * r34 * np.exp(2 * phi_s) * np.exp(2 * phi_f)

    # r_sub = 1 / (1 - r34*r23*np.exp(2*1j*n_sub_ * omega * d_list[1] / c_thz))

    t_sim = t_enu / t_den

    t_sim_abs_ = np.abs(t_sim)  # - 0.0021
    t_sim_abs = np.array([freq_axis_, t_sim_abs_], dtype=float).T

    return t_sim_abs


def transmission_tmm_fit():
    measurement = film_image.get_measurement(*film_eval_pt)
    if sample_idx == 0:
        fit_res_ = film_image.tmm_film_fit(measurement, freq_range=(0.25, 1.20), s_param=9.0)
    else:
        fit_res_ = film_image.tmm_film_fit(measurement, freq_range=(0.25, 3.10), s_param=9.0)

    plt.figure("Tmm fit RI")
    plt.plot(fit_res_["n"][:, 0].real, fit_res_["n"][:, 1].real)
    plt.plot(fit_res_["n"][:, 0].real, fit_res_["n"][:, 1].imag)

    plt.figure("Tmm fit sigma")
    plt.plot(fit_res_["sigma"][:, 0].real, fit_res_["sigma"][:, 1].real)
    plt.plot(fit_res_["sigma"][:, 0].real, fit_res_["sigma"][:, 1].imag)

    return fit_res_


def transmission_drude(freq_axis_, sigma0_, tau_d_, tau_, **kwargs):
    sigma0_ = 1e5 * sigma0_  # 1/(mOhm cm) (1/(1e-3*1e-2)) = 1e5 -> S / m
    sigma = sigma0_ * 1j * tau_d_ / (omega + 1j * tau_d_)

    d_list = array([np.inf, 0.070, 0.0002, np.inf], dtype=float)  # in mm

    lam_vac = c_thz / freq_axis_

    t_tmm_ = np.zeros_like(freq_axis_, dtype=complex)
    for f_idx_ in freq_axis_idx:
        n_film = (1 + 1j) * np.sqrt(sigma / (2 * epsilon_0 * omega * 1e12))
        # kappa = np.linspace(0, )
        # n_sub[:, 1] = (1.7 + 1j*0.05) * np.ones_like(freq_axis_)

        n = array([1, n_sub[f_idx_, 1], n_film[f_idx_], 1], dtype=complex)

        alph_scat = (1 / d_list[1]) * ((n_sub[f_idx_, 1] - 1) * 4 * pi * tau_ / lam_vac[f_idx_])
        ampl_att_ = np.exp(-alph_scat * d_list[1])

        t_tmm_[f_idx_] = coh_tmm("s", n, d_list, angle_in, lam_vac[f_idx_]) * ampl_att_

    freq_axis_ = freq_axis_[freq_axis_idx]
    t_tmm_abs_ = np.abs(t_tmm_[freq_axis_idx])

    plt.figure("Spectrum")
    plt.plot(ref_fd[freq_axis_idx, 0], 20 * np.log10(np.abs(ref_fd[freq_axis_idx, 1]) * t_tmm_abs_))

    t_tmm_abs = np.array([freq_axis_, t_tmm_abs_], dtype=float).T

    return t_tmm_abs


def transmission(sigma0_, tau_, inc_=False, *args):
    freq_axis_ = n_sub[:, 0].real

    sigma0_ = 1e5 * sigma0_  # 1/(mOhm cm) (1/(1e-3*1e-2)) = 1e5 -> S / m
    d_list = array([np.inf, 0.070, 0.0002, np.inf], dtype=float)  # in mm
    c_list = ["i", "i", "i", "i"]

    # n_sub[:, 1].imag = 0.11
    # n_sub[:, 1].real = 1.7

    lam_vac = c_thz / freq_axis_

    t_tmm_ = np.zeros_like(freq_axis_, dtype=complex)
    for f_idx_ in freq_axis_idx:
        if (freq_axis_[f_idx_] > 1.15) and inc_:
            # sigma0_ = 1e5 * 2.16
            # tau_ = 0.0123
            n_film = (1 + 1j) * np.sqrt(sigma0_ / (4 * pi * epsilon_0 * freq_axis_ * 1e12))
            n = array([1, n_sub[f_idx_, 1], n_film[f_idx_], 1], dtype=complex)

            alph_scat = (1 / d_list[1]) * ((n_sub[f_idx_, 1] - 1) * 4 * pi * tau_ / lam_vac[f_idx_])
            ampl_att_ = np.exp(-alph_scat * d_list[1])

            t_tmm_[f_idx_] = np.sqrt(inc_tmm("s", n, d_list, c_list, angle_in, lam_vac[f_idx_])["T"]) * ampl_att_
        else:

            n_film = (1 + 1j) * np.sqrt(sigma0_ / (4 * pi * epsilon_0 * freq_axis_ * 1e12))
            n = array([1, n_sub[f_idx_, 1], n_film[f_idx_], 1], dtype=complex)

            alph_scat = (1 / d_list[1]) * ((n_sub[f_idx_, 1] - 1) * 4 * pi * tau_ / lam_vac[f_idx_])
            ampl_att_ = np.exp(-alph_scat * d_list[1])

            t_tmm_[f_idx_] = coh_tmm("s", n, d_list, angle_in, lam_vac[f_idx_]) * ampl_att_

    # """
    plt.figure("Film refractive index")
    plt.title("Film refractive index")
    plt.plot(freq_axis_, n_film.real, label="RIdx real")
    plt.plot(freq_axis_, n_film.imag, label="RIdx imag")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")
    # """

    freq_axis_ = freq_axis_[freq_axis_idx]
    t_tmm_abs_ = np.abs(t_tmm_[freq_axis_idx])

    plt.figure("Spectrum")
    plt.plot(ref_fd[freq_axis_idx, 0], 20 * np.log10(np.abs(ref_fd[freq_axis_idx, 1]) * t_tmm_abs_))

    t_tmm_abs = np.array([freq_axis_, t_tmm_abs_], dtype=float).T

    return t_tmm_abs


# fit_res = transmission_tmm_fit()

model = transmission_simple
# model = transmission
# model = transmission_drude
inc = False

fig, ax = plt.subplots()
# vals0 = model(freq_axis, init_sigma0, init_tau, inc_=inc)
vals0 = model(freq_axis, init_sigma0, init_tau)
line0, = ax.plot(vals0[:, 0].real, vals0[:, 1], label="TMM + Rayleigh", lw=2, color="blue")
ax.scatter(t_abs_meas[:, 0], t_abs_meas[:, 1], label="Measured", color="red", s=2)
# ax.plot(fit_res["t_abs"][:, 0].real, fit_res["t_abs"][:, 1].real, label="TMM fit")
# ax.scatter(t_abs_meas_sub[:, 0], t_abs_meas_sub[:, 1], label="Measured sub.", color="red", s=2)
# ax.scatter(t_abs_meas_sub[:, 0], t_abs_meas[:, 1] / t_abs_meas_sub[:, 1], label="Diff", color="orange", s=2)
ax.set_ylabel("Amplitude transmission")
ax.set_xlabel("Frequency (THz)")
ax.legend()

fig.subplots_adjust(left=0.25, bottom=0.25)
ax_sigma0 = fig.add_axes([0.25, 0.10, 0.65, 0.03])
if sample_idx == 0:
    val_min, val_max = 0, 140
else:
    val_min, val_max = 0, 10

sigma0_slider = Slider(
    ax=ax_sigma0,
    label=r"$\sigma_{0}$ $(m \Omega cm)^{-1}$",
    valmin=val_min,
    valmax=val_max,
    valinit=init_sigma0,
    orientation="horizontal"
)

ax_tau = fig.add_axes([0.25, 0.15, 0.65, 0.03])
tau_slider = Slider(
    ax=ax_tau,
    label=r"$\tau$ $(um)$",
    valmin=0,
    valmax=150,
    valinit=init_tau,
    orientation="horizontal"
)

ax_tau_d = fig.add_axes([0.05, 0.15, 0.03, 0.70])
tau_d_slider = Slider(
    ax=ax_tau_d,
    label=r"$\tau$ $(THz)$",
    valmin=0,
    valmax=50,
    valinit=init_tau_d_,
    orientation="vertical"
)


def update(val):
    # new_vals = model(freq_axis, sigma0_slider.val, tau_slider.val, inc=inc)
    new_vals = model(freq_axis, sigma0_slider.val, tau_slider.val)
    line0.set_ydata(new_vals[:, 1].real)

    fig.canvas.draw_idle()

    return new_vals[:, 1].real


sigma0_slider.on_changed(update)
tau_slider.on_changed(update)
tau_d_slider.on_changed(update)

resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="0.975")

export_ax = fig.add_axes([0.7, 0.025, 0.1, 0.04])
export_button = Button(export_ax, "Export", hovercolor="0.375")


def export_data(event):
    vals = update(None)
    arr = np.zeros((len(t_abs_meas[:, 0]), 3), dtype=float)
    arr[:, :2] = t_abs_meas.real
    arr[:, 2] = vals
    file_name = f"amplitude_transmission_s{sample_idx + 1}"
    if np.isclose(tau_slider.val, 0):
        file_name += "_noScat"
    save_array_as_csv(arr, file_name)


def reset(event):
    sigma0_slider.reset()
    tau_slider.reset()
    tau_d_slider.reset()


button.on_clicked(reset)
export_button.on_clicked(export_data)

if __name__ == '__main__':
    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            plt.legend()

    plt.show()
