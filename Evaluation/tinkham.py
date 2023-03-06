from consts import *
from Measurements.image import Image
from Measurements.measurements import get_all_measurements
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0


def tinkham(y_sub_fd, y_sam_fd):
    # ???
    d = 200 * 10**-9
    n1, n3 = 1, 1.857

    s = (1 / d) * epsilon_0 * c * (n1 - n3) * (y_sub_fd[:, 1] - y_sam_fd[:, 1]) / y_sam_fd[:, 1]

    s = np.array([y_sub_fd[:, 0], s]).T

    return s


def thin_film(y_sub_fd, y_sam_fd):
    freqs = y_sub_fd[:, 0].real * THz
    omega = 2*pi*freqs

    d = 70 * 10**-6
    n = 1.857
    na, nb = n + 1, n - 1
    Z0 = 377
    delta = omega*d*n/c
    e = np.exp(-1j*delta)

    T_film = y_sam_fd[:, 1] / y_sub_fd[:, 1]

    sigma_enum = na**2 - nb**2 * e + (nb**2 * e - na**2) * T_film
    sigma_denum = (na + nb**2*e) * Z0 * T_film

    s = sigma_enum / sigma_denum

    s = np.array([freqs, s]).T

    return s


if __name__ == '__main__':
    sample_names = ["5x5cm_sqrd", "10x10cm_sqrd_s1", "10x10cm_sqrd_s2", "10x10cm_sqrd_s3"]

    dir_s1_uncoated = data_dir / "Uncoated" / sample_names[1]
    dir_s1_coated = data_dir / "Coated" / sample_names[1]

    measurements = get_all_measurements(data_dir_=dir_s1_uncoated)
    image = Image(measurements)
    s1_uncoated_td, s1_uncoated_fd = image.get_point(x=19, y=9, sub_offset=True, both=True)

    measurements = get_all_measurements(data_dir_=dir_s1_coated)
    image = Image(measurements)
    s1_coated_td, s1_coated_fd = image.get_point(x=19, y=9, sub_offset=True, both=True)

    s = thin_film(s1_uncoated_fd, s1_coated_fd)

    plt.plot(s[:, 0].real, np.abs(s[:, 1]))
    plt.show()
