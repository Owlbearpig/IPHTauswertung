import numpy as np
from numpy import array
from numpy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt
from scipy import signal
from tmm import list_snell, interface_t, interface_r, make_2x2_array


def remove_offset(data_td):
    ret_data_td = data_td.copy()
    offset = (np.mean(ret_data_td[:10, 1]) + np.mean(ret_data_td[-10:, 1])) / 2
    ret_data_td[:, 1] -= offset

    return ret_data_td


def window(data_td, win_len=None, shift=None, en_plot=False, slope=0.15):
    t, y = data_td[:, 0], data_td[:, 1]
    dt = np.mean(np.diff(t))
    win_len = int(win_len / dt)

    if win_len > len(y):
        win_len = len(y)

    win_center = np.argmax(np.abs(y))
    win_start = win_center - int(win_len / 2)

    if win_start < 0:
        win_start = 0

    pre_pad = np.zeros(win_start)
    window_arr = signal.windows.tukey(win_len, slope)
    post_pad = np.zeros(len(y) - win_len - win_start)

    window_arr = np.concatenate((pre_pad, window_arr, post_pad))

    if shift is not None:
        window_arr = np.roll(window_arr, int(shift / dt))

    y_win = y * window_arr

    if en_plot:
        plt.figure("Windowing")
        plt.plot(t, y, label="Before windowing")
        plt.plot(t, np.max(np.abs(y)) * window_arr, label="Window")
        plt.plot(t, y_win, label="After windowing")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (nA)")
        plt.legend()

    return np.array([t, y_win]).T


def do_fft(data_td, en_plots=False):
    dt = float(np.mean(np.diff(data_td[:, 0])))
    # using rfft since measurement is purely real.
    # rfft defined with negative exponent -> np.conj to ensure positive phase slope
    freqs, y = rfftfreq(n=len(data_td[:, 0]), d=dt), np.conj(rfft(data_td[:, 1]))

    if en_plots:
        phi = np.angle(y)
        plt.figure("Phase")
        plt.plot(freqs, phi, label="Phase")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Phase (rad)")
        plt.legend()

        y_db = 20 * np.log10(np.abs(y))
        plt.figure("Spectrum")
        plt.plot(freqs, y_db, label="Magnitude")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Amplitude (dB)")
        plt.legend()
        plt.show()

    return array([freqs, y]).T


def do_ifft(data_fd):
    f_axis, y_fd = data_fd[:, 0].real, data_fd[:, 1]

    y_td = irfft(np.conj(y_fd))
    df = np.mean(np.diff(f_axis))
    n = len(y_td)
    t = np.arange(0, n) / (n * df)

    return array([t, y_td]).T


def f_axis_idx_map(freqs, freq_range=None):
    if freq_range is None:
        freq_range = (0.15, 4.00)
        f0_idx = int(np.argmin(np.abs(freqs - freq_range[0])))
        f1_idx = int(np.argmin(np.abs(freqs - freq_range[1])))
        f_idx = np.arange(f0_idx, f1_idx + 1)
    elif isinstance(freq_range, tuple):
        f0_idx = int(np.argmin(np.abs(freqs - freq_range[0])))
        f1_idx = int(np.argmin(np.abs(freqs - freq_range[1])))
        f_idx = np.arange(f0_idx, f1_idx + 1)
    else:
        single_freq = freq_range
        f_idx = np.array([int(np.argmin(np.abs(freqs - single_freq)))])

    return f_idx


def coh_tmm(pol, n_list, d_list, th_0, lam_vac):
    # Convert lists to numpy arrays if they're not already.
    n_list = array(n_list, dtype=complex)
    d_list = array(d_list, dtype=float)

    num_layers = n_list.size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = list_snell(n_list, th_0)

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    kz_list = 2 * np.pi * n_list * np.cos(th_list) / lam_vac

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = np.seterr(invalid='ignore')
    delta = kz_list * d_list
    np.seterr(**olderr)

    # t_list[i,j] and r_list[i,j] are transmission and reflection amplitudes,
    # respectively, coming from i, going to j. Only need to calculate this when
    # j=i+1. (2D array is overkill but helps avoid confusion.)
    t_list = np.zeros((num_layers, num_layers), dtype=complex)
    r_list = np.zeros((num_layers, num_layers), dtype=complex)
    for i in range(num_layers - 1):
        t_list[i, i + 1] = interface_t(pol, n_list[i], n_list[i + 1],
                                       th_list[i], th_list[i + 1])
        r_list[i, i + 1] = interface_r(pol, n_list[i], n_list[i + 1],
                                       th_list[i], th_list[i + 1])
    # At the interface between the (n-1)st and nth material, let v_n be the
    # amplitude of the wave on the nth side heading forwards (away from the
    # boundary), and let w_n be the amplitude on the nth side heading backwards
    # (towards the boundary). Then (v_n,w_n) = M_n (v_{n+1},w_{n+1}). M_n is
    # M_list[n]. M_0 and M_{num_layers-1} are not defined.
    # My M is a bit different than Sernelius's, but Mtilde is the same.
    M_list = np.zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers - 1):
        M_list[i] = (1 / t_list[i, i + 1]) * np.dot(
            make_2x2_array(np.exp(-1j * delta[i]), 0, 0, np.exp(1j * delta[i]),
                           dtype=complex),
            make_2x2_array(1, r_list[i, i + 1], r_list[i, i + 1], 1, dtype=complex))
    Mtilde = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers - 1):
        Mtilde = np.dot(Mtilde, M_list[i])
    Mtilde = np.dot(make_2x2_array(1, r_list[0, 1], r_list[0, 1], 1,
                                   dtype=complex) / t_list[0, 1], Mtilde)

    # Net complex transmission
    t = 1 / Mtilde[0, 0]

    return t



