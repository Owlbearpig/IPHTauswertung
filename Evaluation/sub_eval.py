from consts import *
from Measurements.image import Image
from Measurements.measurements import get_all_measurements
import matplotlib.pyplot as plt



def phase_extraction(y_fd):
    freqs = y_fd[:, 0].real
    phi = np.unwrap(np.angle(y_fd[:, 1]))
    fit_range = (0.4 < freqs) * (freqs < 0.9)

    z = np.polyfit(y_fd[fit_range, 0].real, phi[fit_range], 1)

    phi_shifted = np.abs(phi - z[1])

    plt.figure("Phase correction")
    plt.plot(freqs, phi_shifted, label="shifted phase")
    plt.plot(freqs, -1*freqs*z[0], label="fit")
    plt.legend()

    return phi_shifted


def analytical_eval(sam_fd, ref_fd):
    d = 0.070 # mm
    freqs = ref_fd[:, 0].real
    omega = 2 * pi * freqs
    phi_s = phase_extraction(sam_fd)
    phi_r = phase_extraction(ref_fd)

    n = 1 + (phi_s - phi_r) * c_thz / (omega * d)

    alpha = -(2/(d/10)) * np.log(np.abs(sam_fd[:, 1] / ref_fd[:, 1])*(n+1)**2/(4*n))

    freq_range = (0.2 < freqs) * (freqs < 2.5)

    plt.figure("RI")
    plt.plot(ref_fd[freq_range, 0].real, n[freq_range])

    plt.figure("alpha")
    plt.plot(ref_fd[freq_range, 0].real, alpha[freq_range])

    return n, alpha

if __name__ == '__main__':
    sample_names = ["5x5cm_sqrd", "10x10cm_sqrd_s1", "10x10cm_sqrd_s2", "10x10cm_sqrd_s3"]

    dir_s1_uncoated = data_dir / "Uncoated" / sample_names[3]
    dir_s1_coated = data_dir / "Coated" / sample_names[3]

    measurements = get_all_measurements(data_dir_=dir_s1_uncoated)

    image = Image(measurements)
    image.plot_image(img_extent=None)
    image.plot_point(x=30, y=2)
    sam_td, sam_fd = image.get_point(x=31, y=12, both=True)
    ref_td, ref_fd = image.get_ref(both=True, coords=(31, 12))

    n, k = analytical_eval(sam_fd, ref_fd)

    plt.show()