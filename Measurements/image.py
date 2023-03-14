import itertools
import random
import re
from consts import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from functions import do_fft, do_ifft, phase_correction, unwrap, window
from measurements import get_all_measurements
from tmm import coh_tmm
from scipy.optimize import shgo
from Evaluation.sub_eval_tmm_numerical import tmm_eval


class Image:
    plotted_ref = False
    noise_floor = None
    time_axis = None
    cache_path = None
    sample_idx = None
    all_points = None
    name = ""

    def __init__(self, data_path, sub_image=None):
        self.data_path = data_path
        self.sub_image = sub_image

        self.refs, self.sams = self._set_measurements()
        self.image_info = self._set_info()
        self.image_data = self._image_cache()

    def _set_measurements(self):
        all_measurements = get_all_measurements(data_dir_=self.data_path)
        refs, sams = self._filter_measurements(all_measurements)

        return refs, sams

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
        parts = self.sams[0].filepath.parts
        self.sample_idx = sample_names.index(parts[-2])
        self.name = f"Sample {self.sample_idx + 1} {parts[-3]}"

        sample_data_td = self.sams[0].get_data_td()
        samples = int(sample_data_td.shape[0])
        self.time_axis = sample_data_td[:, 0].real

        sample_data_fd = self.sams[0].get_data_fd()
        self.freq_axis = sample_data_fd[:, 0].real

        dt = np.mean(np.diff(self.time_axis))

        x_coords, y_coords = [], []
        for sam_measurement in self.sams:
            x_coords.append(sam_measurement.position[0])
            y_coords.append(sam_measurement.position[1])

        x_coords, y_coords = array(sorted(set(x_coords))), array(sorted(set(y_coords)))

        self.all_points = list(itertools.product(x_coords, y_coords))

        w, h = len(x_coords), len(y_coords)
        x_diff, y_diff = np.abs(np.diff(x_coords)), np.abs(np.diff(y_coords))
        dx = np.min(x_diff[np.nonzero(x_diff)])
        dy = np.min(y_diff[np.nonzero(y_diff)])

        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        return {"w": w, "h": h, "dx": dx, "dy": dy, "dt": dt, "samples": samples, "extent": extent}

    def _image_cache(self):
        """
        read all measurements into array and save as npy at location of first measurement
        """
        self.cache_path = Path(self.sams[0].filepath.parent / "cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)

        try:
            img_data = np.load(str(self.cache_path / "_raw_img_cache.npy"))
        except FileNotFoundError:
            w, h, samples = self.image_info["w"], self.image_info["h"], self.image_info["samples"]
            dx, dy = self.image_info["dx"], self.image_info["dy"]
            img_data = np.zeros((w, h, samples))
            min_x, max_x, min_y, max_y = self.image_info["extent"]

            for sam_measurement in self.sams:
                x_pos, y_pos = sam_measurement.position
                x_idx, y_idx = int((x_pos - min_x) / dx), int((y_pos - min_y) / dy)
                img_data[x_idx, y_idx] = sam_measurement.get_data_td(get_raw=True)[:, 1]

            np.save(str(self.cache_path / "_raw_img_cache.npy"), img_data)

        return img_data

    def _coords_to_idx(self, x, y):
        x_idx = int((x - self.image_info["extent"][0]) / self.image_info["dx"])
        y_idx = int((y - self.image_info["extent"][2]) / self.image_info["dy"])

        return x_idx, y_idx

    def _eval_conductivity(self, measurement, selected_freq_):
        point = measurement.position
        d_film = sample_thicknesses[self.sample_idx]
        d_list = [inf, d_sub, d_film, inf]

        film_td = measurement.get_data_td()
        film_ref_td = self.get_ref(both=False, coords=point)

        film_td = window(film_td, win_len=12, shift=0, en_plot=False, slope=0.05)
        film_ref_td = window(film_ref_td, win_len=12, shift=0, en_plot=False, slope=0.05)

        film_ref_fd, film_fd = do_fft(film_ref_td), do_fft(film_td)

        film_ref_fd = phase_correction(film_ref_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)
        film_fd = phase_correction(film_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)

        freqs = film_ref_fd[:, 0].real
        omega = 2 * pi * freqs
        f_idx = np.argmin(np.abs(freqs - selected_freq_))

        try:
            sub_file = list((ROOT_DIR / "Evaluation").glob(f"**/n_sub_s{sample_idx + 1}*.npy"))[1]
            sub_file = ROOT_DIR / "Evaluation" / "n_sub_s3_15.0_18.0.npy"
            n_sub = np.load(sub_file)
        except IndexError:
            position = measurement.position
            n_sub = tmm_eval(self.sub_image, position)
            np.save(ROOT_DIR / "Evaluation" / f"n_sub_s{self.sample_idx + 1}_{position[0]}_{position[1]}.npy", n_sub)

        n_sub = n_sub[f_idx]
        print(f"Substrate refractive index: {np.round(n_sub, 3)}")

        phase_shift = np.exp(-1j * (d_sub + d_film) * omega / c_thz)

        def cost(p):
            n = array([1, n_sub, p[0] + 1j * p[1], 1])
            lam_vac = c_thz / freqs[f_idx]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)["t"]
            sam_tmm_fd = t_tmm_fd * film_ref_fd[f_idx, 1] * phase_shift[f_idx]

            amp_loss = (np.abs(sam_tmm_fd) - np.abs(film_fd[f_idx, 1])) ** 2
            phi_loss = (np.angle(sam_tmm_fd) - np.angle(film_fd[f_idx, 1])) ** 2

            return amp_loss + phi_loss

        bounds = shgo_bounds_film[self.sample_idx]
        iters = shgo_iters
        res = shgo(cost, bounds=bounds, iters=iters-2)
        while (res.fun > 1e-5) and (point[0] < 55):
            iters += 1
            res = shgo(cost, bounds=bounds, iters=iters)
            if iters >= 7:
                break

        n_opt = res.x[0] + 1j * res.x[1]

        epsilon = n_opt ** 2
        sigma = 1j * (1 - epsilon) * epsilon_0 * omega[f_idx] * THz

        print(f"Result: {np.round(int(sigma) * 10 ** -6, 3)} (MS/m), "
              f"n: {np.round(n_opt, 3)}, "
              f"loss: {res.fun}, \n")

        return sigma

    def _calc_grid_vals(self, quantity="p2p", selected_freq=0.800):
        info = self.image_info

        grid_vals_cache_name = self.cache_path / f"{quantity}_{selected_freq}_s3_15.0_18.0.npy"

        if isinstance(selected_freq, tuple) and (quantity in ["MeanConductivity", "ConductivityRange"]):
            try:
                grid_vals = np.load(str(grid_vals_cache_name))
            except FileNotFoundError:
                freq_slice = (selected_freq[0] < self.freq_axis) * (self.freq_axis < selected_freq[1])
                freq_cnt = np.sum(freq_slice)

                grid_vals = np.zeros((info["w"], info["h"], freq_cnt), dtype=complex)
                for f_idx, freq in enumerate(self.freq_axis[freq_slice]):
                    for i, measurement in enumerate(self.sams):
                        print(f"{round(100 * i / len(self.sams), 2)} % done, Frequency: {f_idx} / {freq_cnt}. "
                              f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                        x_idx, y_idx = self._coords_to_idx(*measurement.position)
                        val = self._eval_conductivity(measurement, freq)
                        grid_vals[x_idx, y_idx, f_idx] = val

                np.save(str(grid_vals_cache_name), grid_vals)

            if quantity == "MeanConductivity":
                return np.mean(grid_vals.real, axis=2)
            elif quantity == "ConductivityRange":
                return grid_vals.real

        if quantity == "p2p":
            grid_vals = np.max(self.image_data, axis=2) - np.min(self.image_data, axis=2)
        elif quantity == "Conductivity":
            try:
                grid_vals = np.load(str(grid_vals_cache_name))
            except FileNotFoundError:
                grid_vals = np.zeros((info["w"], info["h"]), dtype=complex)

                for i, measurement in enumerate(self.sams):
                    print(f"{round(100 * i / len(self.sams), 2)} % done. "
                          f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                    x_idx, y_idx = self._coords_to_idx(*measurement.position)
                    val = self._eval_conductivity(measurement, selected_freq)
                    grid_vals[x_idx, y_idx] = val

                np.save(str(grid_vals_cache_name), grid_vals)
        else:
            # grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)
            grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)

        return grid_vals.real

    def plot_image(self, selected_freq=0.800, quantity="p2p", img_extent=None):
        if quantity == "p2p":
            label = ""
        elif quantity == "Conductivity":
            label = " (S/m)  at " + str(np.round(selected_freq, 3)) + " THz"
        else:
            label = " (S/m)"

        grid_vals = self._calc_grid_vals(quantity=quantity, selected_freq=selected_freq)

        info = self.image_info
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        grid_vals = grid_vals[w0:w1, h0:h1]

        fig = plt.figure(f"{self.name} {sample_labels[self.sample_idx]}")
        ax = fig.add_subplot(111)
        ax.set_title(f"{self.name} {sample_labels[self.sample_idx]}")
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.image_info["extent"]

        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                        origin="lower",
                        cmap=plt.get_cmap('jet'),
                        extent=img_extent)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        def fmt(x, val):
            a, b = '{:.2e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        cbar = fig.colorbar(img, format=ticker.FuncFormatter(fmt))
        cbar.set_label(f"{quantity}" + label, rotation=270, labelpad=30)

    def get_point(self, x, y, normalize=False, sub_offset=False, both=False, add_plot=False):
        dx, dy, dt = self.image_info["dx"], self.image_info["dy"], self.image_info["dt"]

        x_idx, y_idx = self._coords_to_idx(x, y)
        y_ = self.image_data[x_idx, y_idx]

        if sub_offset:
            y_ -= np.mean(y_)

        if normalize:
            y_ *= 1 / np.max(y_)

        y_td = np.array([np.linspace(0, dt * len(y_), len(y_)), y_]).T

        if add_plot:
            self.plot_point(x, y, y_td)

        if not both:
            return y_td
        else:
            return y_td, do_fft(y_td)

    def get_ref(self, both=False, normalize=False, sub_offset=False, coords=None):
        if coords is not None:
            closest_sam, best_fit_val = None, np.inf
            for sam_meas in self.sams:
                val = abs(sam_meas.position[0] - coords[0]) + \
                      abs(sam_meas.position[1] - coords[1])
                if val < best_fit_val:
                    best_fit_val = val
                    closest_sam = sam_meas

            closest_ref, best_fit_val = None, np.inf
            for ref_meas in self.refs:
                val = np.abs((closest_sam.meas_time - ref_meas.meas_time).total_seconds())
                if val < best_fit_val:
                    best_fit_val = val
                    closest_ref = ref_meas
            print(f"Time between ref and sample: {(closest_sam.meas_time - closest_ref.meas_time).total_seconds()}")
            ref_td = closest_ref.get_data_td()
        else:
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

    def plot_point(self, x, y, sam_td=None, sub_noise_floor=False, label="", td_scale=1):
        if sam_td is None:
            sam_td = self.get_point(x, y, sub_offset=True)
        ref_td = self.get_ref(sub_offset=True, coords=(x, y))
        # y_td = filtering(y_td, wn=(2.000, 3.000), filt_type="bandpass", order=5)

        sam_td = window(sam_td, win_len=12, shift=0, en_plot=False, slope=0.05)
        ref_td = window(ref_td, win_len=12, shift=0, en_plot=False, slope=0.05)

        ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

        sam_td, sam_fd = phase_correction(sam_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)
        ref_td, ref_fd = phase_correction(ref_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)

        phi_ref, phi_sam = unwrap(ref_fd), unwrap(sam_fd)

        noise_floor = np.mean(20 * np.log10(np.abs(ref_fd[ref_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        if not self.plotted_ref:
            plt.figure("Spectrum")
            plt.plot(ref_fd[plot_range1, 0], 20 * np.log10(np.abs(ref_fd[plot_range1, 1])) - noise_floor,
                     label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Amplitude (dB)")

            plt.figure("Phase")
            plt.plot(ref_fd[plot_range1, 0], phi_ref[plot_range1, 1], label="Reference")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("Phase (rad)")

            plt.figure("Time domain")
            plt.plot(ref_td[:, 0], ref_td[:, 1], label="Reference")
            plt.xlabel("Time (ps)")
            plt.ylabel("Amplitude (Arb. u.)")

            self.plotted_ref = True

        label += f" (x={x} (mm), y={y} (mm))"
        noise_floor = np.mean(20 * np.log10(np.abs(sam_fd[sam_fd[:, 0] > 6.0, 1]))) * sub_noise_floor

        plt.figure("Spectrum")
        plt.plot(sam_fd[plot_range1, 0], 20 * np.log10(np.abs(sam_fd[plot_range1, 1])) - noise_floor, label=label)

        plt.figure("Phase")
        plt.plot(sam_fd[plot_range1, 0], phi_sam[plot_range1, 1], label=label)

        plt.figure("Time domain")
        plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=label + f" (Amplitude x {td_scale})")

    def histogram(self):
        grid_vals = self._calc_grid_vals(quantity="Conductivity", selected_freq=0.800)

        plt.hist(grid_vals, density=True, bins=40)  # density=False would make counts
        plt.ylabel("Conductivity (S/m)")
        plt.xlabel("Count")


if __name__ == '__main__':
    sample_idx = 0

    meas_dir_sub = data_dir / "Uncoated" / sample_names[sample_idx]
    sub_image = Image(data_path=meas_dir_sub)

    meas_dir = data_dir / "Coated" / sample_names[sample_idx]
    film_image = Image(data_path=meas_dir, sub_image=sub_image)
    # s1, s2, s3 = [-10, 50, -3, 27]
    film_image.plot_image(img_extent=[-10, 50, -3, 27], quantity="Conductivity", selected_freq=0.800)

    # s4 = [18, 51, 0, 20]
    # film_image.plot_image(img_extent=[18, 51, 0, 20], quantity="Conductivity", selected_freq=0.800)

    for fig_label in plt.get_figlabels():
        if "Sample" in fig_label:
            continue
        plt.figure(fig_label)
        plt.legend()

    plt.show()
