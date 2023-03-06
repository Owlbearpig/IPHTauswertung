import re
from consts import *
import numpy as np
import matplotlib.pyplot as plt
from functions import do_fft
from measurements import Measurement, get_all_measurements


class Image:
    plotted_ref = False
    time_axis = None
    name = ""

    def __init__(self, measurements):
        self.all_measurements = measurements
        self.refs, self.sams = self._filter_measurements(self.all_measurements)
        self.image_info = self._set_info()
        self.image_data = self._image_cache()

    @staticmethod
    def _filter_measurements(measurements):
        refs, sams = [], []
        for measurement in measurements:
            if measurement.meas_type == "ref":
                refs.append(measurement)
            else:
                sams.append(measurement)

        return refs, sams

    def _set_info(self):
        sample_data = self.sams[0].get_data_td()
        samples = int(sample_data.shape[0])
        self.time_axis = sample_data[:, 0]

        dt = np.mean(np.diff(self.time_axis))

        x_coords, y_coords = [], []
        for sam_measurement in self.sams:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        x_coords, y_coords = sorted(set(x_coords)), sorted(set(y_coords))

        w, h = len(x_coords), len(y_coords)
        dx, dy = np.abs(x_coords[0] - x_coords[1]), np.abs(y_coords[0] - y_coords[1])

        extent = [min(x_coords), max(x_coords), min(y_coords), max(y_coords)]

        parts = self.sams[0].filepath.parts
        self.name = f"{parts[-2]} {parts[-3]}"

        return {"w": w, "h": h, "dx": dx, "dy": dy, "dt": dt, "samples": samples, "extent": extent}

    def _image_cache(self):
        """
        read all measurements into array and save as npy at location of first measurement
        """
        cache_path = self.sams[0].filepath.parent / "_img_cache.npy"

        try:
            img_data = np.load(str(cache_path))
        except FileNotFoundError:
            w, h, samples = self.image_info["w"], self.image_info["h"], self.image_info["samples"]
            dx, dy = self.image_info["dx"], self.image_info["dy"]
            img_data = np.zeros((w, h, samples))
            min_x, max_x, min_y, max_y = self.image_info["extent"]

            for sam_measurement in self.sams:
                x_pos, y_pos = sam_measurement.position
                x_idx, y_idx = int((x_pos - min_x) / dx), int((y_pos - min_y) / dy)
                img_data[x_idx, y_idx] = sam_measurement.get_data_td(get_raw=True)[:, 1]

            np.save(str(cache_path), img_data)

        return img_data

    def plot_image(self, plot_type_="p2p", img_extent=None):
        info = self.image_info
        if plot_type_ == "p2p":
            grid_vals = np.max(self.image_data, axis=2) - np.min(self.image_data, axis=2)
        else:
            # grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)
            grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17/info["dt"]):int(20/info["dt"])]), axis=2)

        if img_extent is None:
            img_extent = info["extent"]
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        grid_vals = grid_vals[w0:w1, h0:h1]

        fig = plt.figure(f"{self.name}")
        ax = fig.add_subplot(111)
        ax.set_title(f"{self.name}")
        fig.subplots_adjust(left=0.2)

        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                        origin="upper",
                        cmap=plt.get_cmap('jet'),
                        extent=img_extent)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        cbar = fig.colorbar(img)
        cbar.set_label(f"{plot_type_}", rotation=270, labelpad=10)

    def get_point(self, x, y, normalize=False, sub_offset=False, both=False):
        dx, dy, dt = self.image_info["dx"], self.image_info["dy"], self.image_info["dt"]
        h = self.image_info["h"]

        x_idx, y_idx = int((x - self.image_info["extent"][0]) / dx), int((y - self.image_info["extent"][2]) / dy)

        y_ = self.image_data[x_idx, h - y_idx]

        if sub_offset:
            y_ -= np.mean(y_)

        if normalize:
            y_ *= 1 / np.max(y_)

        y_td = np.array([np.linspace(0, dt * len(y_), len(y_)), y_]).T

        if not both:
            return y_td
        else:
            return y_td, do_fft(y_td)

    def get_ref(self, both=False, normalize=False, sub_offset=False):
        ref_td = self.refs[-1].get_data_td()

        if sub_offset:
            ref_td[:, 1] -= np.mean(ref_td[:, 1])

        if normalize:
            ref_td[:, 1] *= 1 / np.max(ref_td[:, 1])

        ref_td[:, 0] -= ref_td[0, 0]

        if both:
            ref_fd = do_fft(ref_td)
            return ref_td, ref_fd
        else:
            return ref_td

    def plot_point(self, x, y):
        y_td = self.get_point(x, y, sub_offset=True)

        # y_td = filtering(y_td, wn=(2.000, 3.000), filt_type="bandpass", order=5)
        y_fd = do_fft(y_td)
        ref_td, ref_fd = self.get_ref(both=True, sub_offset=True)

        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1])))

        if not self.plotted_ref:
            plt.figure("Spectrum")
            plt.plot(ref_fd[:, 0], 20 * np.log10(np.abs(ref_fd[:, 1])) - noise_floor, label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Time domain")
            plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (Arb. u.)")

            self.plotted_ref = True

        noise_floor = np.mean(20 * np.log10(np.abs(y_fd[y_fd[:, 0] > 6.0, 1])))

        plt.figure("Spectrum")
        plt.plot(y_fd[:, 0], 20*np.log10(np.abs(y_fd[:, 1])) - noise_floor, label=f"x={x} (mm), y={y} (mm)")
        plt.legend()

        plt.figure("Time domain")
        plt.plot(y_td[:, 0], y_td[:, 1], label=f"x={x} (mm), y={y} (mm)")
        plt.legend()


if __name__ == '__main__':
    sample_names = ["5x5cm_sqrd", "10x10cm_sqrd_s1", "10x10cm_sqrd_s2", "10x10cm_sqrd_s3"]

    dir_s1_uncoated = data_dir / "Uncoated" / sample_names[1]
    dir_s1_coated = data_dir / "Coated" / sample_names[1]

    measurements = get_all_measurements(data_dir_=dir_s1_uncoated)
    image = Image(measurements)
    image.plot_image(img_extent=[0, 40, 0, 20])
    image.plot_point(x=19, y=9)

    measurements = get_all_measurements(data_dir_=dir_s1_coated)
    image = Image(measurements)
    image.plot_image(img_extent=[0, 40, 0, 20])
    image.plot_point(x=19, y=9)

    plt.show()
