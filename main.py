from Measurements.measurements import select_measurements
from Plotting.plot_data import plot_field, plot_absorbance
from Plotting.p2p_image import p2p_image
from Plotting.misc_plot_functions import plot_system_stability
import matplotlib.pyplot as plt
import numpy as np


def main():
    #keywords = ["5x5mm_sqrd"]  # first measurement
    #keywords = ["5x5mm_sqrd_rotated180deg"]  # rotated 180 degree
    #keywords = ["10x10mm_sqrd_1"]
    #keywords = ["10x10mm_sqrd_2"]
    keywords = ["10x10mm_sqrd_3"]

    refs, sams = select_measurements(keywords, match_exact=True)

    p2p_image(refs, sams, point_value="integrated_intensity")
    plt.show()

    """
    matched_sam = None
    for sam in sams:
        if (int(sam.position[0]) == 30) and (int(sam.position[1]) == 20):
            matched_sam = sam
            break

    matched_ref_idx = np.argmin([np.abs(matched_sam.meas_time - ref_i.meas_time) for ref_i in refs])
    matched_ref = refs[matched_ref_idx]
    matched_sam_fd, ref_fd = matched_sam.get_data_fd(), matched_ref.get_data_fd()

    sam_label, ref_label = "sample ", "ref "
    sam_label += f"X={round(matched_sam.position[0], 1)} mm, Y={round(matched_sam.position[1], 1)} mm"
    ref_label += f"X={round(matched_ref.position[0], 1)} mm, Y={round(matched_ref.position[1], 1)} mm"

    plot_field(ref_fd, label=ref_label, freq_range=(0, 6))
    plot_field(matched_sam_fd, label=sam_label, freq_range=(0, 6))

    plot_system_stability(refs)

    plot_absorbance(matched_sam_fd, ref_fd, freq_range=(0.25, 3), label=sam_label)

    """
    sam1500 = sams[1500]
    matched_ref_idx = np.argmin([np.abs(sam1500.meas_time - ref_i.meas_time) for ref_i in refs])
    matched_ref = refs[matched_ref_idx]
    sam1500_fd, ref_fd = sam1500.get_data_fd(), matched_ref.get_data_fd()

    sam_label, ref_label = "sample ", "ref "
    sam_label += f"X={round(sam1500.position[0], 1)} mm, Y={round(sam1500.position[1], 1)} mm"
    ref_label += f"X={round(matched_ref.position[0], 1)} mm, Y={round(matched_ref.position[1], 1)} mm"

    plot_field(ref_fd, label=ref_label, freq_range=(0, 6))
    plot_field(sam1500_fd, label=sam_label, freq_range=(0, 6))
    
    plot_system_stability(refs)

    plot_absorbance(sam1500_fd, ref_fd, freq_range=(0.25, 3), label=sam_label)
    




    """
    plt.figure("System stability")
    t, amp_1THz, pulse_position, p2p_ref = [], [], [], []
    for i in range(1, len(refs)):
        #matched_ref_idx = np.argmin([np.abs(sams[i].meas_time - ref_i.meas_time) for ref_i in refs])
        #matched_ref = refs[matched_ref_idx].get_data_fd()
        #sam_i = sams[i].get_data_fd()
        #amp_1THz.append(np.abs(sam_i[100, 1] / matched_ref[100, 1]))
        ref_fd = refs[i].get_data_fd()
        ref_td = refs[i].get_data_td()
        #amp_1THz.append(np.abs(ref_fd[64, 1]))
        p2p_ref.append(np.abs(np.max(ref_td[:, 1]) - np.min(ref_td[:, 1])))
        #argmax_ref = np.argmax(np.abs(ref_td[:, 1]))
        #pulse_position.append(ref_td[argmax_ref, 0])
        t.append(refs[i].meas_time)
        #plot_field(ref_i, label=f"ref{i}")
    t0 = sorted(t)[0]
    dt = [(ti-t0).total_seconds() / 60 for ti in t]
    plt.plot(dt, p2p_ref, label="p2p reference")
    """




if __name__ == '__main__':
    main()
    plt.show()
