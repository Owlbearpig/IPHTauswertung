from Measurements.measurements import select_measurements
from Plotting.plot_data import plot_field, plot_absorbance
from Plotting.p2p_image import p2p_image
import matplotlib.pyplot as plt
import numpy as np


def main():
    keywords = ["5x5mm_sqrd"]
    refs, sams = select_measurements(keywords)

    p2p_image(sams)

    ref0_fd = refs[0].get_data_fd()
    sam0_fd = sams[0].get_data_fd()

    #plot_field(ref0_fd, label="ref")
    #plot_field(sam0_fd, label="sample0")

    plt.figure("System stability")
    t, amp_1THz = [], []
    for i in range(1, len(refs)):
        #if (i % 20) != 0:
        #    continue

        ref_i = refs[i].get_data_fd()
        amp_1THz.append(np.abs(ref_i[100, 1]))
        t.append(refs[i].meas_time)
        #plot_field(ref_i, label=f"ref{i}")
    plt.plot(t, amp_1THz, label="1 THz amplitude")
    plt.legend()

    plot_absorbance(sam0_fd, ref0_fd, freq_range=(0.4, 2))

    plt.show()


if __name__ == '__main__':
    main()



