from consts import *
import matplotlib.pyplot as plt
from functions import phase_correction


def analytical_eval(sam_fd, ref_fd):
    d = 0.070  # mm
    freqs = ref_fd[:, 0].real
    omega = 2 * pi * freqs
    phi_s = phase_correction(sam_fd, verbose=False)
    phi_r = phase_correction(ref_fd, verbose=False)

    n_real = 1 + (phi_s - phi_r) * c_thz / (omega * d)

    alpha = -(2/d) * np.log(np.abs(sam_fd[:, 1] / ref_fd[:, 1])*(n_real+1)**2/(4*n_real))

    k = c_thz*alpha / (2*omega)

    n = n_real + 1j*k
    for idx, val in enumerate(n):
        if np.isnan(val):
            n[idx] = n[150]

    freq_range = (0.0 < freqs) * (freqs < 11)

    plt.figure("RI")
    plt.plot(ref_fd[freq_range, 0].real, n[freq_range].real, label="Analytical")

    plt.figure("Extinction coefficient")
    plt.plot(ref_fd[freq_range, 0].real, n[freq_range].imag, label="Analytical")

    return n


if __name__ == '__main__':
    from Measurements.image import Image
    from Measurements.measurements import get_all_measurements

    sample_names = ["5x5cm_sqrd", "10x10cm_sqrd_s1", "10x10cm_sqrd_s2", "10x10cm_sqrd_s3"]

    dir_s1_uncoated = data_dir / "Uncoated" / sample_names[3]
    dir_s1_coated = data_dir / "Coated" / sample_names[3]

    measurements = get_all_measurements(data_dir_=dir_s1_uncoated)

    image = Image(measurements)
    image.plot_image(img_extent=None)
    image.plot_point(x=30, y=2)
    sam_td, sam_fd = image.get_point(x=31, y=12, both=True)
    ref_td, ref_fd = image.get_ref(both=True, coords=(31, 12))

    n = analytical_eval(sam_fd, ref_fd)

    plt.show()
