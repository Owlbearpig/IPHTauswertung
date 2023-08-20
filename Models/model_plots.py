import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib.widgets import Slider, Button
from scipy.constants import elementary_charge, electron_mass
from Measurements.image import Image
from consts import data_dir

sample_idx = 3

meas_dir_sub = data_dir / "Uncoated" / "s4"
sub_image = Image(data_path=meas_dir_sub, options={"load_mpl_style": False})

meas_dir_film = data_dir / "s4_new_area" / "Image0"
options = {"cbar_min": 1e5, "cbar_max": 3.5e5, "log_scale": False, "color_map": "viridis",
           "load_mpl_style": False, "invert_x": True, "invert_y": True}  # s4

film_image = Image(meas_dir_film, sub_image, sample_idx, options)

sigma_meas = film_image.plot_conductivity_spectrum(14, -5, freq_range=(0.75, 2.6), smoothen=True)


def drude(f_, tau_, n_, m_eff_, *args):
    tau = tau_ * 1e-15
    n = n_ * 1e6  # n in SI: cm^-3 -> m^-3
    m_eff = m_eff_ * electron_mass
    omega = 2 * pi * f_ * 1e12
    q = elementary_charge

    sigma0 = n * q ** 2 * tau / m_eff
    sigma_t0 = sigma0 / (1 + omega ** 2 * tau ** 2)
    sigma_t1 = 1j * omega * tau * sigma_t0

    sigma = sigma_t0 + sigma_t1

    return sigma


def drude_smith(f_, tau_, n_, m_eff_, c, *args):
    tau = tau_ * 1e-15
    n = n_ * 1e6  # n in SI: cm^-3 -> m^-3
    m_eff = m_eff_ * electron_mass
    omega = 2 * pi * f_ * 1e12
    q = elementary_charge

    sigma0 = n * q ** 2 * tau / m_eff
    f1 = sigma0 / (1 - 1j * omega * tau)
    f2 = 1 + c / (1 - 1j * omega * tau)
    sigma = f1 * f2

    return sigma


model = drude_smith

f = np.linspace(0.1, 3.5, 1000)

init_tau = 80  # tau in fs
init_n = 4.08e16  # in cm^-3
# init_m_eff = 0.46  # in electron mass
init_m_eff = 0.05  # in electron mass
init_c = 1

fig, ax = plt.subplots()
vals0 = model(f, init_tau, init_n, init_m_eff, init_c)
line0, = ax.plot(f, vals0.real, label="Sigma real", lw=2, color="blue")
line1, = ax.plot(f, vals0.imag, label="Sigma imag", lw=2, color="red")

ax.scatter(sigma_meas[:, 0].real, sigma_meas[:, 1].real, label="Sigma real measured", color="blue", s=1)
ax.scatter(sigma_meas[:, 0].real, sigma_meas[:, 1].imag, label="Sigma imag measured", color="red", s=1)

ax.set_ylabel("Conductivity [S/m]")
ax.legend()

ax.set_xlabel('Frequency [THz]')
fig.subplots_adjust(left=0.25, bottom=0.25)

ax_tau = fig.add_axes([0.25, 0.1, 0.65, 0.03])
tau_slider = Slider(
    ax=ax_tau,
    label='Tau [fs]',
    valmin=1,
    valmax=500,
    valinit=init_tau,
    orientation="horizontal"
)

ax_n = fig.add_axes([0.25, 0.15, 0.65, 0.03])
n_slider = Slider(
    ax=ax_n,
    label="Carrier concentration [$cm^{-3}$]",
    valmin=1.00e16,
    valmax=1.00e18,
    valinit=init_n,
    orientation="horizontal"
)

ax_m = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
m_slider = Slider(
    ax=ax_m,
    label="Carrier mass [$m_0$]",
    valmin=0,
    valmax=0.1,
    valinit=init_m_eff,
    orientation="vertical"
)

ax_c = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
c_slider = Slider(
    ax=ax_c,
    label="c",
    valmin=-1,
    valmax=1,
    valinit=init_c,
    orientation="vertical"
)


def update(val):
    new_vals = model(f, tau_slider.val, n_slider.val, m_slider.val, c_slider.val)
    sig_real, sig_imag = new_vals.real, new_vals.imag
    line0.set_ydata(new_vals.real)
    line1.set_ydata(new_vals.imag)

    sig_real_min, sig_real_max = np.min(sig_real), np.max(sig_real)
    sig_imag_min, sig_imag_max = np.min(sig_real), np.max(sig_real)

    y_min = 0.9 * min(sig_real_min, sig_real_max)
    y_max = 1.1 * max(sig_imag_min, sig_imag_max)

    ax.set_ylim((y_min, y_max))

    fig.canvas.draw_idle()


tau_slider.on_changed(update)
n_slider.on_changed(update)
m_slider.on_changed(update)
c_slider.on_changed(update)

resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    tau_slider.reset()
    n_slider.reset()
    m_slider.reset()
    c_slider.reset()


button.on_clicked(reset)

if __name__ == '__main__':
    plt.show()
