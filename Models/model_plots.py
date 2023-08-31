import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib.widgets import Slider, Button
from scipy.constants import elementary_charge, electron_mass
from Measurements.image import Image
from consts import data_dir
from mpl_settings import mpl_style_params

mpl_style_params()

sample_idx = 3

meas_dir_sub = data_dir / "Uncoated" / "s4"
sub_image = Image(data_path=meas_dir_sub, options={"load_mpl_style": False})

meas_dir_film = data_dir / "s4_new_area" / "Image0"
options = {"cbar_min": 1e5, "cbar_max": 3.5e5, "log_scale": False, "color_map": "viridis",
           "load_mpl_style": False, "invert_x": True, "invert_y": True}  # s4

film_image = Image(meas_dir_film, sub_image, sample_idx, options)

eval_point = (10, -5)
save_file = f"sigma_x{eval_point[0]}_y{eval_point[1]}.npy"
try:
    sigma_meas = np.load(save_file)
except FileNotFoundError:
    sigma_meas = film_image.plot_conductivity_spectrum(*eval_point, freq_range=(0.10, 2.5), smoothen=False)
    sigma_meas[:, 1] *= 0.01
    np.save(save_file, sigma_meas)


def drude(f_, tau_, n_, m_eff_, *args):
    tau = tau_
    n = n_ * 1e6  # n in SI: cm^-3 -> m^-3
    m_eff = m_eff_ * electron_mass
    omega = 2 * pi * f_
    q = elementary_charge

    sigma0 = n * q ** 2 * tau / m_eff
    sigma_t0 = sigma0 / (1 + omega ** 2 * tau ** 2)
    sigma_t1 = 1j * omega * tau * sigma_t0

    sigma = sigma_t0 + sigma_t1

    return sigma


# def drude_smith(f_, tau_, n_, m_eff_, c, *args):
def drude_smith(f_, tau_, sigma0, c, *args):
    tau = tau_ * 1e3
    # n = n_ * 1e6  # n in SI: cm^-3 -> m^-3
    # m_eff = m_eff_ * electron_mass
    omega = 2 * pi * f_
    q = elementary_charge

    # sigma0 = n * q ** 2 * tau / m_eff
    f1 = sigma0 / (1 - 1j * omega * tau)
    f2 = 1 + c / (1 - 1j * omega * tau)
    sigma = f1 * f2

    return sigma


def ssftc_not(f_, tau_, n_, m_eff_, omega_p_, *args):
    tau = tau_
    n = n_ * 1e6  # n in SI: cm^-3 -> m^-3
    m_eff = m_eff_ * electron_mass
    omega = 2 * pi * f_
    omega_p = omega_p_
    q = elementary_charge

    f1 = n * q ** 2 / m_eff
    den = 1 - (1j * tau * (omega - omega_p ** 2 / omega))

    return f1 * tau / den


def ssftc(freq_, f_, tau_, s0_, tau_t_, s0_t_, *args):
    tau = tau_ * 1e-3  # tau_ in fs -> ps

    tau_t = tau_t_  # ps
    omega = 2 * pi * freq_

    s_f = s0_ / (1 - 1j * omega * tau)

    s_t = s0_t_ * (-1j * omega * tau_t) / np.log(1 - 1j * omega * tau_t)

    s1 = f_ / s_f
    s2 = (1 - f_) / s_t

    return 1 / (s1 + s2)


model = ssftc

f = np.linspace(0.1, 3.5, 1000)
f = np.logspace(11, 13, 1000) / 1e12

init_tau = 20  # tau in fs
init_tau_t = 0.4  # tau_t in ps
# init_n = 4.08e16  # in cm^-3
# init_m_eff = 0.46  # in electron mass
# init_m_eff = 0.10  # in electron mass
# init_c = 1
init_f = 0.84
init_s0 = 1000  # 100
init_s0_t = 100  # 5.7

fig, ax = plt.subplots()
vals0 = model(f, init_f, init_tau, init_s0, init_tau_t, init_s0_t)
# vals0 = model(f, init_tau, init_s0, init_f)
# vals0 = model(f, init_tau, init_s0)
line0, = ax.plot(f, vals0.real, label="Sigma real", lw=2, color="blue")
line1, = ax.plot(f, vals0.imag, label="Sigma imag", lw=2, color="red")

y_min = -2500  # -60
y_max = 5000  # 120
#ax.set_ylim((y_min, y_max))

ax.scatter(sigma_meas[:, 0].real, sigma_meas[:, 1].real, label="Sigma real measured", color="blue", s=1)
ax.scatter(sigma_meas[:, 0].real, sigma_meas[:, 1].imag, label="Sigma imag measured", color="red", s=1)

ax.set_ylabel("Conductivity ($\Omega^{-1} cm^{-1}$)")
ax.legend()

ax.set_xlabel('Frequency (THz)')
fig.subplots_adjust(left=0.25, bottom=0.25)

ax_f = fig.add_axes([0.25, 0.05, 0.65, 0.03])
f_slider = Slider(
    ax=ax_f,
    label='f',
    valmin=0,
    valmax=1,
    valinit=init_f,
    orientation="horizontal"
)

ax_tau = fig.add_axes([0.25, 0.1, 0.65, 0.03])
tau_slider = Slider(
    ax=ax_tau,
    label='Tau [fs]',
    valmin=1,
    valmax=10000,
    valinit=init_tau,
    orientation="horizontal"
)

ax_tau_t = fig.add_axes([0.25, 0.15, 0.65, 0.03])
tau_slider_t = Slider(
    ax=ax_tau_t,
    label='Tau_t [ps]',
    valmin=0.1,
    valmax=500,
    valinit=init_tau_t,
    orientation="horizontal"
)

ax_s0 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
s0_slider = Slider(
    ax=ax_s0,
    label="$\sigma_{0}$",
    valmin=1,
    valmax=10000,
    valinit=init_s0,
    orientation="vertical"
)

ax_s0_t = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
s0_slider_t = Slider(
    ax=ax_s0_t,
    label="$\sigma_{0,t}$",
    valmin=1,
    valmax=5000,
    valinit=init_s0_t,
    orientation="vertical"
)


def update(val):
    new_vals = model(f, f_slider.val, tau_slider.val, s0_slider.val, tau_slider_t.val, s0_slider_t.val)
    # new_vals = model(f, tau_slider.val, s0_slider.val, f_slider.val)
    # new_vals = model(f, tau_slider.val, s0_slider.val)
    sig_real, sig_imag = new_vals.real, new_vals.imag
    line0.set_ydata(new_vals.real)
    line1.set_ydata(new_vals.imag)

    sig_real_min, sig_real_max = np.min(sig_real), np.max(sig_real)
    sig_imag_min, sig_imag_max = np.min(sig_imag), np.max(sig_imag)

    ax.set_ylim((y_min, y_max))

    fig.canvas.draw_idle()


f_slider.on_changed(update)
tau_slider.on_changed(update)
tau_slider_t.on_changed(update)

s0_slider.on_changed(update)
s0_slider_t.on_changed(update)

resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    tau_slider.reset()
    tau_slider_t.reset()
    f_slider.reset()
    s0_slider.reset()
    s0_slider_t.reset()


button.on_clicked(reset)

if __name__ == '__main__':
    plt.show()
