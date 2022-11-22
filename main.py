from Measurements.measurements import select_measurements
from Plotting.plot_data import plot_field
import matplotlib.pyplot as plt


def main():
    keywords = ["5x5mm_sqrd"]
    refs, sams = select_measurements(keywords)

    ref0_fd = refs[0].get_data_fd()
    sam0_fd = sams[0].get_data_fd()

    plot_field(ref0_fd, label="ref")
    plot_field(sam0_fd, label="sample0")
    for i in range(1, 10):
        sam_i = sams[i].get_data_fd()
        plot_field(sam_i, label=f"sample{i}")

    plt.show()


if __name__ == '__main__':
    main()



