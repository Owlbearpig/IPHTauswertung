from Measurements.measurements import select_measurements
from Plotting.plot_data import plot_field, plot_absorbance
from Plotting.p2p_image import p2p_image
import matplotlib.pyplot as plt
import numpy as np


def main():
    keywords = ["5x5mm_sqrd"]
    refs, sams = select_measurements(keywords)

    #p2p_image(refs, sams)

    sam1500 = sams[1500]
    matched_ref_idx = np.argmin([np.abs(sam1500.meas_time - ref_i.meas_time) for ref_i in refs])
    matched_ref = refs[matched_ref_idx]
    sam1500_fd, ref_fd = sam1500.get_data_fd(), matched_ref.get_data_fd()

    sam_label, ref_label = "sample ", "ref "
    sam_label += f"X={round(sam1500.position[0], 1)} mm, Y={round(sam1500.position[1], 1)} mm"
    ref_label += f"X={round(matched_ref.position[0], 1)} mm, Y={round(matched_ref.position[1], 1)} mm"

    plot_field(ref_fd, label=ref_label, freq_range=(0, 6))
    plot_field(sam1500_fd, label=sam_label, freq_range=(0, 6))

    plt.figure("System stability")
    t, amp_1THz = [], []
    for i in range(1, len(refs)):
        #matched_ref_idx = np.argmin([np.abs(sams[i].meas_time - ref_i.meas_time) for ref_i in refs])
        #matched_ref = refs[matched_ref_idx].get_data_fd()
        #sam_i = sams[i].get_data_fd()
        #amp_1THz.append(np.abs(sam_i[100, 1] / matched_ref[100, 1]))
        ref_fd = refs[i].get_data_fd()
        amp_1THz.append(np.abs(ref_fd[100, 1]))
        t.append(refs[i].meas_time)
        #plot_field(ref_i, label=f"ref{i}")
    plt.plot(t, amp_1THz, label="1 THz amplitude")
    plt.legend()

    plot_absorbance(sam1500_fd, ref_fd, freq_range=(0.25, 3), label=sam_label)

    plt.show()


if __name__ == '__main__':
    main()



