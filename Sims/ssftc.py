import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from consts import ROOT_DIR
from numpy import pi
from scipy.constants import epsilon_0



# sigma_meas = np.load(ROOT_DIR / "Models" / "sigma_x10_y-5.npy")
sigma_meas = np.load(ROOT_DIR / "Models" / "sigma_x5_y-2.npy")
sigma_meas[:, 1] *= 0.01

freqs_meas = 1e12*sigma_meas[:, 0].real
freqs_meas_log = np.log10(freqs_meas)

freqs = np.logspace(0, 15, 1000)
omega = 2 * np.pi * freqs
log_omega = np.log10(omega)

"""
init_sigma0_d = 150  # S 1/cm
init_t = 0.02  # ps
init_sigma0_t = 28  # S 1/cm
init_tt = 0.4  # ps
init_f = 0.83  # factor

"""
init_sigma0_d = 3700  # S 1/cm
init_t = 50  # fs
init_sigma0_t = 27.96  # S 1/cm
init_tt = 29.7  # ps
init_f = 0.002  # factor


def lorentz_factor(t_, n=1):
    c = [-1, *n*[0]]
    s = 1
    for idx in range(1, n+1):
        s += c[idx-1] / ((1-1j*omega*t_) ** idx)

    return s


def ssftc(sigma0_d_, t_, sigma0_t_, tt_, f):
    t_ *= 1e-15
    tt_ *= 1e-12

    lf = lorentz_factor(t_, n=1)
    sigma_f = sigma0_d_ / (1 - 1j * omega * t_)
    sigma_f *= lf

    sigma_t_ = -1j * sigma0_t_ * omega * tt_ / np.log(1 - 1j * omega * tt_)

    sigma_eff = 1 / (f / sigma_f + (1 - f) / sigma_t_)

    return sigma_eff


# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
ax.plot(freqs_meas, sigma_meas[:, 1].real, label="Measured real")
ax.plot(freqs_meas, sigma_meas[:, 1].imag, label="Measured imag")

init_val = ssftc(init_sigma0_d, init_t, init_sigma0_t, init_tt, init_f)
line_real, = ax.plot(freqs, init_val.real, lw=2, c="blue")
line_imag, = ax.plot(freqs, init_val.imag, lw=2, c="red")
ax.set_xlabel(r'Frequency (Hz)')
ax.set_ylabel('Conductivity (S/cm)')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

ax_sigma_dc = fig.add_axes([0.25, 0.1, 0.65, 0.03])
sigma_dc_slider = Slider(
    ax=ax_sigma_dc,
    label='$\sigma_{dc}$ (S 1/cm)',
    valmin=1000,
    valmax=5000,
    valinit=init_sigma0_d,
    orientation="horizontal"
)

ax_t = fig.add_axes([0.25, 0.05, 0.65, 0.03])
t_slider = Slider(
    ax=ax_t,
    label=r'$\tau$ (fs)',
    valmin=1,
    valmax=100,
    valinit=init_t,
    orientation="horizontal"
)

ax_f = fig.add_axes([0.25, 0.15, 0.65, 0.03])
f_slider = Slider(
    ax=ax_f,
    label=r'f',
    valmin=0,
    valmax=1,
    valinit=init_f,
    orientation="horizontal"
)

ax_sigma_t = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
sigma_t_slider = Slider(
    ax=ax_sigma_t,
    label=r'$\sigma_{t}$ (S 1/cm)',
    valmin=1,
    valmax=2000,
    valinit=init_sigma0_t,
    orientation="vertical"
)

ax_tt = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
tt_slider = Slider(
    ax=ax_tt,
    label=r'$\tau_{t}$ (ps)',
    valmin=1,
    valmax=200,
    valinit=init_tt,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    new_val = ssftc(sigma_dc_slider.val, t_slider.val, sigma_t_slider.val, tt_slider.val, f_slider.val)
    line_real.set_ydata(new_val.real)
    line_imag.set_ydata(new_val.imag)
    fig.canvas.draw_idle()


# register the update function with each slider
sigma_dc_slider.on_changed(update)
t_slider.on_changed(update)
sigma_t_slider.on_changed(update)
tt_slider.on_changed(update)
f_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.02])
button = Button(resetax, 'Reset', hovercolor='0.405')


def reset(event):
    sigma_dc_slider.reset()
    t_slider.reset()
    sigma_t_slider.reset()
    tt_slider.reset()
    f_slider.reset()


button.on_clicked(reset)
ax.legend()
plt.show()
