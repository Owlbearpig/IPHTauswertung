from consts import data_dir
from Measurements.image import Image
from functions import do_fft, window
import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt
from scipy.constants import c as c0

"""
# what's the attenuation roughly ? with beer law 
"""

sample_idx = 3

meas_dir_sub = data_dir / "Uncoated" / "s3"
sub_image = Image(data_path=meas_dir_sub)

meas_dir = data_dir / "s4_new_area" / "Image0"
options = {"cbar_min": 1e5, "cbar_max": 3.5e5, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": True}  # s4

film_image = Image(meas_dir, sub_image, sample_idx, options)


def kappa_init():
    point = (14, -5)
    point = (10, 10)

    film_measurement = film_image.get_measurement(*point)
    film_td = film_measurement.get_data_td()
    film_td[:, 0] -= film_td[0, 0]

    t = film_td[:, 0]

    sub_measurement = sub_image.get_measurement(*point)
    sub_td = sub_measurement.get_data_td()
    ref_td = sub_image.get_ref(both=False, coords=point)

    ref_fd = do_fft(ref_td)
    sub_fd, film_fd = do_fft(sub_td), do_fft(film_td)

    f = ref_fd[:, 0].real

    max_ref, max_sam = np.max(ref_td[:, 1]), np.max(sub_td[:, 1])
    l = 200 * 1e-9
    l = 40 * 1e-6
    omega = 2 * pi * f * 1e12

    k1 = (-1/l) * (c0 / omega) * np.log(max_sam/max_ref)

    plt.figure("kappa FD")
    plt.plot(f[10:400], k1[10:400])

    plt.figure("Signal TD")
    plt.plot(t, ref_td[:, 1], label="ref")
    plt.plot(t, sub_td[:, 1], label="sub")
    plt.legend()
    plt.show()


def main():
    kappa_init()


if __name__ == '__main__':
    main()
