import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, array
from matplotlib.widgets import Slider, Button
from scipy.constants import elementary_charge, electron_mass, epsilon_0
from Measurements.image import Image
from consts import data_dir, c_thz, angle_in
from mpl_settings import mpl_style_params
from tmm_slim import coh_tmm
from Evaluation.sub_eval_tmm_numerical import tmm_eval
from functions import f_axis_idx_map

mpl_style_params()

sample_idx = 3
film_eval_pt = (10, -5)
sub_eval_pt = (40, 10)  # (37.5, 18.5) # high p2p

meas_dir_sub = data_dir / "Uncoated" / "s4"
sub_image = Image(data_path=meas_dir_sub, options={"load_mpl_style": False})
sub_image.plot_image()
sub_image.plot_transmission(*sub_eval_pt)
n_sub = tmm_eval(sub_image, eval_point_=sub_eval_pt, en_plot=True)

meas_dir_film = data_dir / "s4_new_area" / "Image0"
options = {"cbar_min": 1e5, "cbar_max": 3.5e5, "log_scale": False, "color_map": "viridis",
           "load_mpl_style": False, "invert_x": True, "invert_y": True}  # s4

film_image = Image(meas_dir_film, sub_image, sample_idx, options)

save_file = f"t_abs_x{film_eval_pt[0]}_y{film_eval_pt[1]}.npy"

data_range = slice(10, 300)
try:
    t_abs_meas = np.load(save_file)
except FileNotFoundError:
    t_abs, t_abs_meas = film_image.plot_transmittance(*film_eval_pt, freq_range=(0.10, 3.0))
    np.save(save_file, t_abs_meas)

t_abs_meas = t_abs_meas[data_range, :]

freq_axis = n_sub[:, 0].real
freq_axis_idx = f_axis_idx_map(freq_axis, freq_range=(0.10, 3.0))
omega = 2 * pi * freq_axis

init_sigma0 = 3.38  # 1 / (mOhm cm)


def transmission_simple(freq_axis_, sigma0_):
    sigma0_ = 1e5 * sigma0_  # 1/(mOhm cm) (1/(1e-3*1e-2)) = 1e5 -> S / m
    d_list = array([np.inf, 0.090, 0.0002, np.inf], dtype=float)  # in mm
    n_film_ = (1 + 1j) * np.sqrt(sigma0_ / (4 * pi * epsilon_0 * freq_axis_ * 1e12))
    n_sub_ = n_sub[:, 1]

    plt.figure("Film refractive index")
    plt.title("Film refractive index")
    plt.plot(freq_axis_, n_film_.real, label="RIdx real")
    plt.plot(freq_axis_, n_film_.imag, label="RIdx imag")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")

    t12 = 2 * 1 / (n_sub_ + 1)
    r12 = (1 - n_sub_) / (n_sub_ + 1)

    t23 = 2 * n_sub_ / (n_sub_ + n_film_)
    r23 = (n_sub_ - n_film_) / (n_sub_ + n_film_)

    t34 = 2 * n_film_ / (1 + n_film_)
    r34 = (n_film_ - 1) / (1 + n_film_)

    t_enu = t12 * t23 * t34 * np.exp(1j * n_film_ * omega * d_list[2] / c_thz) * np.exp(
        1j * n_sub_ * omega * d_list[1] / c_thz)
    t_den = 1 + r12 * r23 * np.exp(2 * 1j * omega * d_list[2] * n_film_ / c_thz)

    t_sim = t_enu / t_den

    freq_axis_ = freq_axis_[freq_axis_idx]
    t_sim_abs_ = np.abs(t_sim[freq_axis_idx])
    t_sim_abs = np.array([freq_axis_, t_sim_abs_], dtype=float).T

    return t_sim_abs


def transmission(freq_axis_, sigma0_):
    sigma0_ = 1e5 * sigma0_  # 1/(mOhm cm) (1/(1e-3*1e-2)) = 1e5 -> S / m
    d_list = array([np.inf, 0.070, 0.0002, np.inf], dtype=float)  # in mm
    n_film = (1 + 1j) * np.sqrt(sigma0_ / (4 * pi * epsilon_0 * freq_axis_ * 1e12))
    plt.figure("Film refractive index")
    plt.title("Film refractive index")
    plt.plot(freq_axis_, n_film.real, label="RIdx real")
    plt.plot(freq_axis_, n_film.imag, label="RIdx imag")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Refractive index")

    t_tmm_ = np.zeros_like(freq_axis_, dtype=complex)
    for freq_idx_ in freq_axis_idx:
        n = array([1, n_sub[freq_idx_, 1], n_film[freq_idx_], 1], dtype=complex)

        lam_vac = c_thz / freq_axis_[freq_idx_]
        t_tmm_[freq_idx_] = coh_tmm("s", n, d_list, angle_in, lam_vac)

    freq_axis_ = freq_axis_[freq_axis_idx]
    t_tmm_abs_ = np.abs(t_tmm_[freq_axis_idx])
    t_tmm_abs = np.array([freq_axis_, t_tmm_abs_], dtype=float).T

    return t_tmm_abs


fig, ax = plt.subplots()
vals0 = transmission(freq_axis, init_sigma0)
line0, = ax.plot(vals0[:, 0].real, vals0[:, 1], label="TMM", lw=2, color="blue")
ax.scatter(t_abs_meas[:, 0], t_abs_meas[:, 1], label="Measured", color="red", s=2)
ax.set_ylabel("Amplitude transmission")
ax.set_xlabel("Frequency (THz)")
ax.legend()

fig.subplots_adjust(left=0.25, bottom=0.25)
ax_sigma0 = fig.add_axes([0.25, 0.10, 0.65, 0.03])
sigma0_slider = Slider(
    ax=ax_sigma0,
    label=r"$\sigma_{0}$ $(m \Omega cm)^{-1}$",
    valmin=0,
    valmax=20,
    valinit=init_sigma0,
    orientation="horizontal"
)


def update(val):
    new_vals = transmission(freq_axis, sigma0_slider.val)
    line0.set_ydata(new_vals[:, 1].real)

    fig.canvas.draw_idle()


sigma0_slider.on_changed(update)

resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="0.975")


def reset(event):
    sigma0_slider.reset()


button.on_clicked(reset)

if __name__ == '__main__':
    plt.show()
