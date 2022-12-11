from imports import *
from functions import phase_correction, do_ifft, calc_absorption


def plot_field(data_fd, label="", color=None, freq_range=(0, 10)):
    # TODO use measurement instead of fd array
    freq_range = (freq_range[0] <= data_fd[:, 0]) * (data_fd[:, 0] <= freq_range[1])
    freqs = data_fd[freq_range, 0]

    plt.figure("Wrapped phase")
    plt.title("Wrapped phase")
    plt.plot(freqs, np.angle(data_fd[freq_range, 1]), label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    phase_unwrapped = phase_correction(data_fd[freq_range, :])
    plt.figure("Unwrapped phase")
    plt.title("Unwrapped phase")
    plt.plot(freqs, phase_unwrapped, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Phase (rad)")
    plt.legend()

    plt.figure("Spectrum")
    plt.title("Spectrum")
    plt.plot(freqs, 10 * np.log10(np.abs(data_fd[freq_range, 1])), label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude (dB)")
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
        plt.plot(data_fd[freq_range, 0], data_fd[freq_range, 1].real, label=label, color=color)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("real(E)")
        plt.legend()

        plt.figure("Imag part")
        plt.title("Imag part")
        plt.plot(data_fd[freq_range, 0], data_fd[freq_range, 1].imag, label=label, color=color)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Imag(E)")
        plt.legend()


def plot_ri(n, label="", color=None, freq_range=(0, 10)):
    freq_range = (freq_range[0] <= n[:, 0].real) * (n[:, 0].real <= freq_range[1])
    freqs = n[freq_range, 0].real

    plt.figure("Refractive index real")
    plt.plot(freqs, n[freq_range, 1].real, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.xlabel("Refractive index")
    plt.legend()

    plt.figure("Refractive index imag")
    plt.plot(freqs, n[freq_range, 1].imag, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Extinction coefficient")
    plt.legend()

    a = calc_absorption(freqs, n[freq_range, 1].imag)
    plt.figure("Absorption coefficient")
    plt.plot(freqs, a, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Absorption coefficient (1/cm)")
    plt.legend()


def plot_absorbance(sam_data_fd, ref_data_fd, label="", color=None, freq_range=(0, 10)):
    freq_range = (freq_range[0] <= ref_data_fd[:, 0]) * (ref_data_fd[:, 0] <= freq_range[1])
    freqs = ref_data_fd[freq_range, 0]
    absorbance = 10 * np.log10(np.abs(ref_data_fd[freq_range, 1]) / np.abs(sam_data_fd[freq_range, 1]))

    plt.figure("Absorbance")
    plt.title("Absorbance")
    plt.plot(freqs, absorbance, label=label, color=color)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Absorbance (dB)")
    plt.legend()


if __name__ == '__main__':
    pass
