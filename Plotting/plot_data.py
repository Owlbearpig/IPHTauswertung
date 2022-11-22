from imports import *
from functions import phase_correction, do_ifft, calc_absorption


def plot_field(data_fd, label="", color=None):
    freqs = data_fd[:, 0]

    plt.figure("Wrapped phase")
    plt.title("Wrapped phase")
    plt.plot(freqs, np.angle(data_fd[:, 1]), label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    phase_unwrapped = phase_correction(data_fd)
    plt.figure("Unwrapped phase")
    plt.title("Unwrapped phase")
    plt.plot(freqs, phase_unwrapped, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    plt.figure("Spectrum")
    plt.title("Spectrum")
    plt.plot(freqs, np.abs(data_fd[:, 1]), label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()

    data_td = do_ifft(data_fd)

    plt.figure("Time domain")
    plt.title("Time domain")
    plt.plot(data_td[:, 0], data_td[:, 1], label=label, color=color)
    plt.xlabel("Time (ps)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()

    if False:
        plt.figure("Real part")
        plt.title("Real part")
        plt.plot(data_fd[:, 0], data_fd[:, 1].real, label=label, color=color)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("real(E)")
        plt.legend()

        plt.figure("Imag part")
        plt.title("Imag part")
        plt.plot(data_fd[:, 0], data_fd[:, 1].imag, label=label, color=color)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Imag(E)")
        plt.legend()


def plot_ri(n, label="", color=None):
    freqs = n[:, 0].real

    plt.figure("Refractive index real")
    plt.plot(freqs, n[:, 1].real, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.xlabel("Refractive index")
    plt.legend()

    plt.figure("Refractive index imag")
    plt.plot(freqs, n[:, 1].imag, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Extinction coefficient")
    plt.legend()

    a = calc_absorption(freqs, n[:, 1].imag)
    plt.figure("Absorption coefficient")
    plt.plot(freqs, a, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Absorption coefficient (1/cm)")
    plt.legend()


if __name__ == '__main__':
    pass
