from consts import data_dir
from Measurements.image import Image
from functions import do_fft, window
import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt

"""
# how do you correctly unwrap ??
"""

sample_idx = 3

meas_dir_sub = data_dir / "Uncoated" / "s4"
sub_image = Image(data_path=meas_dir_sub)

meas_dir = data_dir / "s4_new_area" / "Image0"
options = {"cbar_min": 1e5, "cbar_max": 3.5e5, "log_scale": False, "color_map": "viridis",
           "invert_x": True, "invert_y": True}  # s4

film_image = Image(meas_dir, sub_image, sample_idx, options)


def unwrap_phase():
    point = (14, -5)
    point = (33.0, 5.5)
    measurement = sub_image.get_measurement(*point)
    sub_image.plot_image()
    sam_td = measurement.get_data_td()
    # sam_td[:, 0] -= sam_td[0, 0]

    t = sam_td[:, 0]

    ref_td = sub_image.get_ref(both=False, coords=point)

    ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)
    # sub_image.plot_point(10, 10)

    f = ref_fd[:, 0].real

    argmax_ref, argmax_sam = np.argmax(ref_td[:, 1]), np.argmax(sam_td[:, 1])

    t0_ref, t0_sam = t[argmax_ref], t[argmax_sam]

    phi0_ref, phi0_sam = 2 * pi * t0_ref * f, 2 * pi * t0_sam * f
    phi0_offset = 2 * pi * (t0_sam - t0_ref) * f

    phi_ref = np.angle(ref_fd[:, 1] * exp(-1j * phi0_ref))
    phi_sam = np.angle(sam_fd[:, 1] * exp(-1j * phi0_sam))

    phi0 = np.unwrap(phi_sam - phi_ref)

    phi = phi0 - phi0_ref + phi0_sam + phi0_offset

    phi = np.angle(np.exp(1j * phi))

    plt.figure("Phase FD")
    plt.plot(f, phi)

    plt.figure("Signal TD")
    plt.plot(t, ref_td[:, 1], label="ref")
    plt.plot(t, sam_td[:, 1], label="sam")
    plt.legend()
    plt.show()


def main():
    unwrap_phase()


if __name__ == '__main__':
    main()
