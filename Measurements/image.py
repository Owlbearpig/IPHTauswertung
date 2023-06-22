import itertools
import random
import re
import timeit
from itertools import product
from consts import *
import numpy as np
import matplotlib.ticker as ticker
from mpl_settings import *
from functions import do_fft, do_ifft, phase_correction, unwrap, window
from measurements import get_all_measurements
from tmm_slim import coh_tmm
# from tmm import coh_tmm
from scipy.optimize import shgo
from Evaluation.sub_eval_tmm_numerical import tmm_eval


class Image:
    plotted_ref = False
    noise_floor = None
    time_axis = None
    cache_path = None
    sample_idx = None
    all_points = None
    options = {}
    name = ""

    def __init__(self, data_path, sub_image=None, sample_idx=None, options=None):
        self.data_path = data_path
        self.sub_image = sub_image

        self.refs, self.sams = self._set_measurements()
        if sample_idx is not None:
            self.sample_idx = sample_idx

        self.image_info = self._set_info()
        self._set_options(options)
        self.image_data = self._image_cache()

    def _set_options(self, options):
        if options is None:
            options = {}

        # set defaults if missing # TODO use default_dict ?
        if "excluded_areas" not in options.keys():
            options["excluded_areas"] = None
        if "one2onesub" not in self.options.keys():
            options["one2onesub"] = False

        if "cbar_min" not in self.options.keys():
            options["cbar_min"] = 0
        if "cbar_max" not in self.options.keys():
            options["cbar_max"] = np.inf

        if "log_scale" not in self.options.keys():
            options["log_scale"] = False

        if "color_map" not in self.options.keys():
            options["color_map"] = "autumn"

        if "invert_x" not in self.options.keys():
            options["invert_x"] = False
        if "invert_y" not in self.options.keys():
            options["invert_y"] = False

        self.options.update(options)

    def _set_measurements(self):
        all_measurements = get_all_measurements(data_dir_=self.data_path)
        refs, sams = self._filter_measurements(all_measurements)

        refs = tuple(sorted(refs, key=lambda meas: meas.meas_time))
        sams = tuple(sorted(sams, key=lambda meas: meas.meas_time))
        first_measurement = min(refs[0], sams[0], key=lambda meas: meas.meas_time)
        print("First measurement at: ", first_measurement.meas_time, "\n")

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
        if self.sample_idx is None:
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

        self._empty_grid = np.zeros((w, h), dtype=complex)

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

    def _idx_to_coords(self, x_idx, y_idx):
        dx, dy = self.image_info["dx"], self.image_info["dy"]
        y = self.image_info["extent"][2] + y_idx * dy
        x = self.image_info["extent"][0] + x_idx * dx

        return x, y

    def _eval_conductivity(self, measurement, selected_freq_, **kwargs):
        point = measurement.position

        if "d_film" not in kwargs.keys():
            d_film = sample_thicknesses[self.sample_idx]
        else:
            d_film = kwargs["d_film"]

        """ # calc mean of two layers sample_idx 2
        if self.sample_idx == 2:
            d_list = [inf, d_sub, *d_film, inf]
        else:
            d_list = [inf, d_sub, d_film, inf]
        """
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

        """
        try:
            sub_file = list((ROOT_DIR / "Evaluation").glob(f"**/n_sub_s{sample_idx + 1}*.npy"))[1]
            n_sub = np.load(sub_file)
        except IndexError:
            position = measurement.position
            n_sub = tmm_eval(self.sub_image, position)
            np.save(ROOT_DIR / "Evaluation" / f"n_sub_s{self.sample_idx + 1}_{position[0]}_{position[1]}.npy", n_sub)
        """

        if self.options["one2onesub"]:
            n_sub = tmm_eval(self.sub_image, point, single_f_idx=f_idx)
        else:
            try:
                sub_file = ROOT_DIR / "Evaluation" / f"n_sub_s{self.sample_idx + 1}_10_10.npy"
                n_sub = np.load(sub_file)
            except FileNotFoundError:
                position = (10, 10)
                n_sub = tmm_eval(self.sub_image, position)
                np.save(ROOT_DIR / "Evaluation" / f"n_sub_s{self.sample_idx + 1}_{position[0]}_{position[1]}.npy",
                        n_sub)
            n_sub = n_sub[f_idx]

        print(f"Substrate refractive index: {np.round(n_sub, 3)}")

        phase_shift = np.exp(-1j * (d_sub + np.sum(d_film)) * omega / c_thz)

        # film_ref_interpol = self._ref_interpolation(measurement, selected_freq_=selected_freq_, ret_cart=True)

        def cost(p):
            """
            if self.sample_idx == 2:
                n = array([1, n_sub, p[0] + 1j * p[1], p[2] + 1j * p[3], 1])
            else:
                n = array([1, n_sub, p[0] + 1j * p[1], 1])
            """
            n = array([1, n_sub, p[0] + 1j * p[1], 1])

            lam_vac = c_thz / freqs[f_idx]
            t_tmm_fd = coh_tmm("s", n, d_list, angle_in, lam_vac)

            sam_tmm_fd = t_tmm_fd * film_ref_fd[f_idx, 1] * phase_shift[f_idx]
            # sam_tmm_fd = t_tmm_fd * film_ref_interpol * phase_shift[f_idx]

            amp_loss = (np.abs(sam_tmm_fd) - np.abs(film_fd[f_idx, 1])) ** 2
            phi_loss = (np.angle(sam_tmm_fd) - np.angle(film_fd[f_idx, 1])) ** 2

            return amp_loss + phi_loss

        if "bounds" not in kwargs.keys():
            bounds = shgo_bounds_film[self.sample_idx]
        else:
            bounds = kwargs["bounds"]
        iters = shgo_iters - 3
        res = shgo(cost, bounds=bounds, iters=iters - 2)
        while (res.fun > 1e-10) and (point[0] < 55):
            iters += 1
            res = shgo(cost, bounds=bounds, iters=iters)
            if iters >= 5:
                break
        """
        if self.sample_idx == 2:
            n_opt = res.x[0] + 1j * res.x[1], res.x[2] + 1j * res.x[3]
            epsilon = n_opt[0] ** 2
            # epsilon = n_opt[1] ** 2
        else:
            n_opt = res.x[0] + 1j * res.x[1]
            epsilon = n_opt ** 2
        """
        n_opt = res.x[0] + 1j * res.x[1]
        epsilon = n_opt ** 2

        sigma = 1j * (1 - epsilon) * epsilon_0 * omega[f_idx] * THz

        print(f"Result: {np.round(sigma * 10 ** -6, 5)} (MS/m), "
              f"n: {np.round(n_opt, 3)}, "
              f"loss: {res.fun}, \n")

        if ("ret_loss" in kwargs.keys()) and kwargs["ret_loss"]:
            return res.fun

        return sigma

    def _calc_power_grid(self, freq_range):
        def power(measurement):
            freq_slice = (freq_range[0] < self.freq_axis) * (self.freq_axis < freq_range[1])

            ref_td, ref_fd = self.get_ref(coords=measurement.position, both=True)

            sam_fd = measurement.get_data_fd()
            power_val_sam = np.sum(np.abs(sam_fd[freq_slice, 1])) / np.sum(freq_slice)
            power_val_ref = np.sum(np.abs(ref_fd[freq_slice, 1])) / np.sum(freq_slice)

            return (power_val_sam / power_val_ref) ** 2

        grid_vals = self._empty_grid.copy()

        for i, measurement in enumerate(self.sams):
            print(f"{round(100 * i / len(self.sams), 2)} % done. "
                  f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
            x_idx, y_idx = self._coords_to_idx(*measurement.position)
            val = power(measurement)
            grid_vals[x_idx, y_idx] = val

        return grid_vals

    def _is_excluded(self, idx_tuple):
        if self.options["excluded_areas"] is None:
            return False

        if np.array(self.options["excluded_areas"]).ndim == 1:
            areas = [self.options["excluded_areas"]]
        else:
            areas = self.options["excluded_areas"]

        for area in areas:
            x, y = self._idx_to_coords(*idx_tuple)
            if (area[0] <= x <= area[1]) * (area[2] <= y <= area[3]):
                return True

        return False

    def _calc_grid_vals(self, quantity="p2p", selected_freq=1.200):
        info = self.image_info

        if self.options["one2onesub"]:
            grid_vals_cache_name = self.cache_path / f"{quantity}_{selected_freq}_s{self.sample_idx + 1}_121sub_layer1.npy"
        else:
            grid_vals_cache_name = self.cache_path / f"{quantity}_{selected_freq}_s{self.sample_idx + 1}_layer1.npy"

        # grid_vals_cache_name = self.cache_path / f"{quantity}_{selected_freq}_s{self.sample_idx + 1}_10_10.npy"

        if isinstance(selected_freq, tuple) and (quantity in ["MeanConductivity", "ConductivityRange"]):
            try:
                grid_vals = np.load(str(grid_vals_cache_name))
            except FileNotFoundError:
                freq_slice = (selected_freq[0] < self.freq_axis) * (self.freq_axis < selected_freq[1])
                freq_cnt = np.sum(freq_slice)

                grid_vals = np.zeros((info["w"], info["h"], freq_cnt), dtype=complex)
                for f_idx, freq in enumerate(self.freq_axis[freq_slice]):
                    for i, measurement in enumerate(self.sams):
                        pos = measurement.position
                        print(f"{round(100 * i / len(self.sams), 2)} % done, Frequency: {f_idx} / {freq_cnt}. "
                              f"(Measurement: {i}/{len(self.sams)}, {pos} mm)")
                        x_idx, y_idx = self._coords_to_idx(*pos)
                        val = self._eval_conductivity(measurement, freq)

                        grid_vals[x_idx, y_idx, f_idx] = val

                np.save(str(grid_vals_cache_name), grid_vals)

            if quantity == "MeanConductivity":
                return np.mean(grid_vals.real, axis=2)
            elif quantity == "ConductivityRange":
                return grid_vals.real
        if quantity.lower() == "power":
            if isinstance(selected_freq, tuple):
                grid_vals = self._calc_power_grid(freq_range=selected_freq)
            else:
                print("Selected frequency must be range given as tuple")
                grid_vals = self._empty_grid
        elif quantity == "p2p":
            grid_vals = np.max(self.image_data, axis=2) - np.min(self.image_data, axis=2)
        elif quantity.lower() == "conductivity":
            try:
                grid_vals = np.load(str(grid_vals_cache_name))
            except FileNotFoundError:
                grid_vals = self._empty_grid.copy()

                for i, measurement in enumerate(self.sams):
                    pos = measurement.position
                    print(f"{round(100 * i / len(self.sams), 2)} % done. "
                          f"(Measurement: {i}/{len(self.sams)}, {pos} mm)")
                    val = self._eval_conductivity(measurement, selected_freq)
                    x_idx, y_idx = self._coords_to_idx(*pos)
                    grid_vals[x_idx, y_idx] = val

                np.save(str(grid_vals_cache_name), grid_vals)
        elif quantity.lower() == "ref_amp":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                amp_, _ = self._ref_interpolation(measurement, selected_freq_=selected_freq,
                                                  ret_cart=False)
                grid_vals[x_idx, y_idx] = amp_
        elif quantity.lower() == "loss":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                loss = self._eval_conductivity(measurement, selected_freq, ret_loss=True)
                grid_vals[x_idx, y_idx] = np.log10(loss)

        elif quantity == "Reference phase":
            grid_vals = self._empty_grid.copy()

            for i, measurement in enumerate(self.sams):
                print(f"{round(100 * i / len(self.sams), 2)} % done. "
                      f"(Measurement: {i}/{len(self.sams)}, {measurement.position} mm)")
                x_idx, y_idx = self._coords_to_idx(*measurement.position)
                _, phi_ = self._ref_interpolation(measurement, selected_freq_=selected_freq,
                                                  ret_cart=False)
                grid_vals[x_idx, y_idx] = phi_
        else:
            # grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)
            grid_vals = np.argmax(np.abs(self.image_data[:, :, int(17 / info["dt"]):int(20 / info["dt"])]), axis=2)

        return grid_vals.real

    def plot_cond_vs_d(self, point=None, freq=None):
        if point is None:
            point = (33.0, 11.0)

        measurement = self.get_measurement(*point)
        if freq is None:
            freq = 1.200

        thicknesses = np.arange(0.000001, 0.001000, 0.000001)
        conductivities = np.zeros_like(thicknesses, dtype=complex)
        for idx, d in enumerate(thicknesses):
            val = self._eval_conductivity(measurement, selected_freq_=freq, d_film=d)
            conductivities[idx] = val

        plt.figure("Conductivity vs thickness")
        plt.title(f"Conductivity as a function of thickness at {freq} THz\n{self.name} {point} mm")
        plt.plot(thicknesses * 1e6, conductivities.real, label="Conductivity real part")
        plt.plot(thicknesses * 1e6, conductivities.imag, label="Conductivity imaginary part")
        plt.xlabel("Thickness (nm)")
        plt.ylabel("Conductivity (S/m)")
        plt.legend()
        plt.show()

    def _exclude_pixels(self, grid_vals):
        filtered_grid = grid_vals.copy()
        dims = filtered_grid.shape
        for x_idx in range(dims[0]):
            for y_idx in range(dims[1]):
                if self._is_excluded((x_idx, y_idx)):
                    filtered_grid[x_idx, y_idx] = 0

        return filtered_grid

    def _mirror_image(self, grid_vals):
        # should flip horizontal axis around center
        grid_vals_mirrored = grid_vals.copy()

        w = grid_vals.shape[0] - 1
        for x_idx in range(grid_vals.shape[0]):
            grid_vals_mirrored[w - x_idx, :] = grid_vals[x_idx, :]

        return grid_vals_mirrored

    def _smoothen_(self, grid_vals):
        return grid_vals
        # TODO
        grid_vals_smooth = grid_vals.copy()
        w, h = grid_vals.shape

        for x_idx in range(w):
            for y_idx in range(h):
                try:
                    if np.sum(grid_vals[x_idx - 1:x_idx + 1, y_idx - 1:y_idx + 1]) > 4 * 1e6:
                        grid_vals_smooth[x_idx:y_idx] = 1e6
                except IndexError as e:
                    print(e)
                    continue

        return grid_vals_smooth

    def plot_image(self, selected_freq=None, quantity="p2p", img_extent=None, flip_x=False):
        if quantity.lower() == "p2p":
            label = ""
        elif quantity.lower() == "conductivity":
            label = " (S/m)  at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity.lower() == "ref_amp":
            label = " Interpolated ref. amp. at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity == "Reference phase":
            label = " interpolated at " + str(np.round(selected_freq, 3)) + " THz"
        elif quantity.lower() == "power":
            label = f"({selected_freq[0]}-{selected_freq[1]}) THz"
        elif quantity.lower() == "loss":
            label = " function value (log10)"
        else:
            label = " (S/m)"

        info = self.image_info
        if img_extent is None:
            w0, w1, h0, h1 = [0, info["w"], 0, info["h"]]
        else:
            dx, dy = info["dx"], info["dy"]
            w0, w1 = int((img_extent[0] - info["extent"][0]) / dx), int((img_extent[1] - info["extent"][0]) / dx)
            h0, h1 = int((img_extent[2] - info["extent"][2]) / dy), int((img_extent[3] - info["extent"][2]) / dy)

        grid_vals = self._calc_grid_vals(quantity=quantity, selected_freq=selected_freq)

        grid_vals = grid_vals[w0:w1, h0:h1]

        grid_vals = self._exclude_pixels(grid_vals)

        if self.options["log_scale"]:
            grid_vals = np.log10(grid_vals)

        if flip_x:
            grid_vals = self._mirror_image(grid_vals)

        grid_vals = self._smoothen_(grid_vals)

        fig = plt.figure(f"{self.name} {sample_labels[self.sample_idx]}")
        ax = fig.add_subplot(111)
        ax.set_title(f"{self.name} {sample_labels[self.sample_idx]}")
        fig.subplots_adjust(left=0.2)

        if img_extent is None:
            img_extent = self.image_info["extent"]

        if self.options["log_scale"]:
            self.options["cbar_min"] = np.log10(self.options["cbar_min"])
            self.options["cbar_max"] = np.log10(self.options["cbar_max"])

        try:
            cbar_min = np.min(grid_vals[grid_vals > self.options["cbar_min"]])
            cbar_max = np.max(grid_vals[grid_vals < self.options["cbar_max"]])
        except ValueError:
            print("Check cbar bounds")
            cbar_min = np.min(grid_vals[grid_vals > 0])
            cbar_max = np.max(grid_vals[grid_vals < np.inf])

        # grid_vals[grid_vals < self.options["cbar_min"]] = 0
        # grid_vals[grid_vals > self.options["cbar_max"]] = 0

        img = ax.imshow(grid_vals.transpose((1, 0)),
                        vmin=cbar_min, vmax=cbar_max,
                        origin="lower",
                        cmap=plt.get_cmap(self.options["color_map"]),
                        extent=img_extent)
        if self.options["invert_x"]:
            ax.invert_xaxis()
        if self.options["invert_y"]:
            ax.invert_yaxis()

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        if np.max(grid_vals) > 1000:
            cbar = fig.colorbar(img, format=ticker.FuncFormatter(fmt))
        else:
            cbar = fig.colorbar(img)

        if quantity.lower() == "conductivity":
            label = " at " + str(np.round(selected_freq, 3)) + " THz"
            cbar.set_label("$\log_{10}$($\sigma$) " + label, rotation=270, labelpad=30)
            cbar.set_label("$\sigma$ (S/m) " + label, rotation=270, labelpad=30)
        else:
            cbar.set_label(f"{quantity}" + label, rotation=270, labelpad=30)

    def get_measurement(self, x, y):
        closest_sam, best_fit_val = None, np.inf
        for sam_meas in self.sams:
            val = abs(sam_meas.position[0] - x) + \
                  abs(sam_meas.position[1] - y)
            if val < best_fit_val:
                best_fit_val = val
                closest_sam = sam_meas

        return closest_sam

    def get_point(self, x, y, normalize=False, sub_offset=False, both=False, add_plot=False):
        dx, dy, dt = self.image_info["dx"], self.image_info["dy"], self.image_info["dt"]

        x_idx, y_idx = self._coords_to_idx(x, y)
        y_ = self.image_data[x_idx, y_idx]

        if sub_offset:
            y_ -= np.mean(y_)

        if normalize:
            y_ *= 1 / np.max(y_)

        t = np.arange(0, len(y_)) * dt
        y_td = np.array([t, y_]).T

        if add_plot:
            self.plot_point(x, y, y_td)

        if not both:
            return y_td
        else:
            return y_td, do_fft(y_td)

    def get_ref(self, both=False, normalize=False, sub_offset=False, coords=None, ret_meas=False):
        if coords is not None:
            closest_sam = self.get_measurement(*coords)

            closest_ref, best_fit_val = None, np.inf
            for ref_meas in self.refs:
                val = np.abs((closest_sam.meas_time - ref_meas.meas_time).total_seconds())
                if val < best_fit_val:
                    best_fit_val = val
                    closest_ref = ref_meas
            print(f"Time between ref and sample: {(closest_sam.meas_time - closest_ref.meas_time).total_seconds()}")
            chosen_ref = closest_ref
        else:
            chosen_ref = self.refs[-1]

        ref_td = chosen_ref.get_data_td()

        if sub_offset:
            ref_td[:, 1] -= np.mean(ref_td[:, 1])

        if normalize:
            ref_td[:, 1] *= 1 / np.max(ref_td[:, 1])

        ref_td[:, 0] -= ref_td[0, 0]

        if ret_meas:
            return chosen_ref

        if both:
            ref_fd = do_fft(ref_td)
            return ref_td, ref_fd
        else:
            return ref_td

    def plot_point(self, x, y, sam_td=None, ref_td=None, sub_noise_floor=False, label="", td_scale=1):
        if (sam_td is None) and (ref_td is None):
            sam_td = self.get_point(x, y, sub_offset=True)
            ref_td = self.get_ref(sub_offset=True, coords=(x, y))

            sam_td = window(sam_td, win_len=25, shift=0, en_plot=False, slope=0.05)
            ref_td = window(ref_td, win_len=25, shift=0, en_plot=False, slope=0.05)

            ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

            sam_td, sam_fd = phase_correction(sam_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)
            ref_td, ref_fd = phase_correction(ref_fd, fit_range=(0.8, 1.6), extrapolate=True, both=True)

        else:
            ref_fd, sam_fd = do_fft(ref_td), do_fft(sam_td)

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
        plt.plot(sam_td[:, 0], td_scale * sam_td[:, 1], label=label + f"\n(Amplitude x {td_scale})")

    def histogram(self):
        grid_vals = self._calc_grid_vals(quantity="Conductivity", selected_freq=0.800)

        plt.hist(grid_vals, density=True, bins=40)  # density=False would make counts
        plt.ylabel("Conductivity (S/m)")
        plt.xlabel("Count")

    def system_stability(self, selected_freq_=0.800):
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))

        ref_ampl_arr, ref_angle_arr = [], []

        t0 = self.refs[0].meas_time
        meas_times = [(ref.meas_time - t0).total_seconds() / 3600 for ref in self.refs]
        for i, ref in enumerate(self.refs):
            ref_td = ref.get_data_td()
            # ref_td = window(ref_td, win_len=12, shift=0, en_plot=False, slope=0.05)
            ref_fd = do_fft(ref_td)
            # ref_fd = phase_correction(ref_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)

            ref_ampl_arr.append(np.sum(np.abs(ref_fd[f_idx, 1])) / 1)
            phi = np.angle(ref_fd[f_idx, 1])
            """ ???
            if i and (abs(ref_angle_arr[-1] - phi) > pi):
                phi -= 2 * pi
            """
            ref_angle_arr.append(phi)
        ref_angle_arr = np.unwrap(ref_angle_arr)
        ref_angle_arr -= np.mean(ref_angle_arr)
        ref_ampl_arr -= np.mean(ref_ampl_arr)

        random.seed(10)
        rnd_sam = random.choice(self.sams)
        position1 = (19, 4)
        position2 = (20, 4)
        sam1 = self.get_measurement(*position1)  # rnd_sam
        sam2 = self.get_measurement(*position2)  # rnd_sam

        sam_t1 = (sam1.meas_time - t0).total_seconds() / 3600
        amp_interpol1, phi_interpol1 = self._ref_interpolation(sam1, ret_cart=False, selected_freq_=selected_freq_)

        sam_t2 = (sam2.meas_time - t0).total_seconds() / 3600
        amp_interpol2, phi_interpol2 = self._ref_interpolation(sam2, ret_cart=False, selected_freq_=selected_freq_)

        plt.figure("System stability amplitude")
        plt.title(f"Reference amplitude at {selected_freq_} THz")
        plt.plot(meas_times, ref_ampl_arr, label=t0)
        # plt.plot(sam_t1, amp_interpol1, marker="o", markersize=5, label=f"Interpol (x={position1[0]}, y={position1[1]}) mm")
        # plt.plot(sam_t2, amp_interpol2, marker="o", markersize=5, label=f"Interpol (x={position2[0]}, y={position2[1]}) mm")
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Amplitude (Arb. u.)")

        plt.figure("System stability angle")
        plt.title(f"Reference phase at {selected_freq_} THz")
        plt.plot(meas_times, ref_angle_arr, label=t0)
        # plt.plot(sam_t1, phi_interpol1, marker="o", markersize=5, label=f"Interpol (x={position1[0]}, y={position1[1]}) mm")
        # plt.plot(sam_t2, phi_interpol2, marker="o", markersize=5, label=f"Interpol (x={position2[0]}, y={position2[1]}) mm")
        plt.xlabel("Measurement time (hour)")
        plt.ylabel("Phase (rad)")

    def _ref_interpolation(self, sam_meas, selected_freq_=0.800, ret_cart=False):
        sam_meas_time = sam_meas.meas_time

        nearest_ref_idx, smallest_time_diff, time_diff = None, np.inf, 0
        for ref_idx in range(len(self.refs)):
            time_diff = (self.refs[ref_idx].meas_time - sam_meas_time).total_seconds()
            if abs(time_diff) < abs(smallest_time_diff):
                nearest_ref_idx = ref_idx
                smallest_time_diff = time_diff

        t0 = self.refs[0].meas_time
        if smallest_time_diff <= 0:
            # sample was measured after reference
            ref_before = self.refs[nearest_ref_idx]
            ref_after = self.refs[nearest_ref_idx + 1]
        else:
            ref_before = self.refs[nearest_ref_idx - 1]
            ref_after = self.refs[nearest_ref_idx]

        t = [(ref_before.meas_time - t0).total_seconds(), (ref_after.meas_time - t0).total_seconds()]
        ref_before_td, ref_after_td = ref_before.get_data_td(), ref_after.get_data_td()

        # ref_before_td = window(ref_before_td, win_len=12, shift=0, en_plot=False, slope=0.05)
        # ref_after_td = window(ref_after_td, win_len=12, shift=0, en_plot=False, slope=0.05)

        ref_before_fd, ref_after_fd = do_fft(ref_before_td), do_fft(ref_after_td)

        # ref_before_fd = phase_correction(ref_before_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)
        # ref_after_fd = phase_correction(ref_after_fd, fit_range=(0.8, 1.6), extrapolate=True, ret_fd=True, en_plot=False)

        # if isinstance(selected_freq_, tuple):

        # else:
        f_idx = np.argmin(np.abs(self.freq_axis - selected_freq_))
        y_amp = [np.sum(np.abs(ref_before_fd[f_idx, 1])) / 1,
                 np.sum(np.abs(ref_after_fd[f_idx, 1])) / 1]
        y_phi = [np.angle(ref_before_fd[f_idx, 1]), np.angle(ref_after_fd[f_idx, 1])]

        amp_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_amp)
        phi_interpol = np.interp((sam_meas_time - t0).total_seconds(), t, y_phi)

        if ret_cart:
            return amp_interpol * np.exp(1j * phi_interpol)
        else:
            return amp_interpol, phi_interpol

    def _4pp_measurement(self, s_idx=0, row_id=None):
        if row_id is None:
            s1 = array([6.49, 6.40, 6.40, 6.64, 6.38, 6.29, 6.38, 6.47, 6.29, 6.22]) * 1e5
            s4 = array([1.09, 1.46, 1.26, 0.81, 1.28, 1.30]) * 1e4
            map_ = {"0": s1, "3": s4}

            return map_[str(s_idx)]

        s2_r2 = array([1.13, 3.09, 7.35, 4.56, 22.6, 2.31, 7.82, 10.9, 13.3, 10.4, 9.13,
                       9.57, 8.54, 10.1, 4.69, 0.682, 12.9, 10.3, 10.7, 10.7, 9.06, 0.937, 0.421]) * 1e3
        s4_r3 = array([2.08, 5.64, 5.44, 3.43, 6.51, 16.6, 10.8, 11.5, 12.7, 9.5, 6.29]) * 1e4

        # map_ = {"s2_r2": s2_r2[:11], "s4_r3": s4_r3}
        map_ = {"s2_r2": s2_r2, "s4_r3": s4_r3}

        if "flipped" not in str(self.data_path):
            slice_ = slice(0, 11)
            vals = map_[f"s{s_idx + 1}_r{row_id}"][slice_]
        else:
            vals = array([10.1, 4.69, 0.682, 12.9, 10.3, 10.7, 10.7, 9.06, 0.937]) * 1e3

        return vals

    def _average_area(self, p0=None, line_len=None, arrow_height=None):
        if p0 is None:
            p0 = (33, 15)  # point upper right (mm)

        if line_len is None:
            s = 3.5  # mm
            line_len = 3 * s

        if arrow_height is None:
            # arrow_height = 0  # self.image_info["dy"] assume single pixel, mm
            arrow_height = 2

        p1 = p0[0] - line_len, p0[1] + arrow_height  # second point of arrow
        print(f"segment: {p0}, {p1}")
        # convert to idx. Direction of the line doesn't matter
        p1_idx, p0_idx = self._coords_to_idx(*p1), self._coords_to_idx(*p0)
        min_x, max_x = min(p1_idx[0], p0_idx[0]), max(p1_idx[0], p0_idx[0])
        min_y, max_y = min(p1_idx[1], p0_idx[1]), max(p1_idx[1], p0_idx[1])

        x_range = range(min_x, max_x + 1)
        y_range = range(min_y, max_y + 1)
        area_indices = list(product(x_range, y_range))

        grid_vals = self._calc_grid_vals(quantity="Conductivity")
        conductivities = [grid_vals[p] for p in area_indices]
        print(conductivities)
        avg_val = np.sum(conductivities) / len(conductivities)
        print(avg_val, "\n")

        return avg_val

    def thz_vs_4pp(self, row_idx, p0=None, en_plot=True):
        def scale(val, dst=None):
            """
            Scale the given value to the scale of dst.
            """
            val = array(val)
            # return val
            # return np.log10(val)

            if dst is None:
                dst = (0, 1)
            src = np.min(val), np.max(val)
            # return val / np.max(val)
            return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]

        _4pp_vals = self._4pp_measurement(s_idx=self.sample_idx, row_id=row_idx)
        _4pp_val_scaled = scale(_4pp_vals)

        if self.sample_idx == 3:
            line_len = 3.5  # s4
        else:
            line_len = 3.5  # 4.0 # s2

        segment_cnt = len(_4pp_vals)
        positions = range(1, segment_cnt + 1)

        if p0 is None:
            # pr_s1 = (33, 15)
            # pr_s4_13 = (2, 13)
            # pr_s4_46 = (30, 12)
            pr_s4_r3 = (37, 6)  # best
            # pr_s4_r3 = (41, 6)
            p0_s2_r2 = (40, 10.5)
            # p0_s2_r2 = (44, 15.5)
            p0 = p0_s2_r2

        thz_cond_vals = {}
        for dy in [0, 1, 2, 4]:
            thz_cond = []
            for i in range(segment_cnt):
                pr = (p0[0] - i * line_len, p0[1])
                avg_val = self._average_area(pr, line_len, dy)
                thz_cond.append(avg_val)

            thz_cond = np.array(thz_cond)
            thz_cond_vals[str(dy)] = scale(thz_cond)

        if "flipped" in str(self.data_path):
            #positions = np.arange(22, 13, -1)
            positions = np.arange(14, 23, 1)

        # thz_cond_vals_scaled = scale(thz_cond_vals)

        if en_plot:
            fig = plt.figure("THz vs 4pp")
            ax = fig.add_subplot(111)
            ax.set_title("$\sigma_{THz}$(1.2 THz) vs $\sigma_{4pp}$(DC)")
            #ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
            ax.scatter(_4pp_vals, thz_cond_vals["0"])
            ax.set_xlabel("$\sigma_{4pp}$(DC) (S/m)")
            ax.set_ylabel("$\sigma_{THz}$(1.2 THz) (S/m)")

            fig = plt.figure("4pp")
            ax = fig.add_subplot(111)
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
            tick_spacing = 1
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

            ax.plot(positions, _4pp_val_scaled, label="$\sigma_{4pp}$(DC)")
            # plt.plot(thz_slope * _4pp_slope)

            ax.set_ylabel("Conductivity (normalized)")
            ax.set_xlabel("Position idx")

            for dy in thz_cond_vals.keys():
                plt.figure("4pp")
                ax.plot(positions, thz_cond_vals[str(dy)], label="$\sigma_{THz}$(1.2 THz)" + f" dy={dy} mm")

        thz_slope = np.sign(np.diff(thz_cond_vals["0"]))
        _4pp_slope = np.sign(np.diff(_4pp_val_scaled))

        return np.sum(thz_slope * _4pp_slope < 0)

    def correlation_image(self, row_idx, p0=None):
        fig = plt.figure("correlation grid")
        ax = fig.add_subplot(111)
        ax.set_title("Summed 4pp and THz anticorrelation")
        if p0 is None:
            x_, y_ = [32, 42], [0, 15]
        else:
            x_, y_ = [int(p0[0]-7), int(p0[0]+7)], [int(p0[1]-7), int(p0[1]+7)]

        corr_grid = np.zeros((x_[1] - x_[0], y_[1] - y_[0]))
        for i, x in enumerate(range(*x_)):
            for j, y in enumerate(range(*y_)):
                corr = film_image.thz_vs_4pp(row_idx=row_idx, p0=(x, y), en_plot=False)
                corr_grid[i, j] = corr
        img = ax.imshow(corr_grid.transpose((1, 0)), extent=[*x_, *y_], origin="lower", vmin=1, vmax=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        cbar = fig.colorbar(img)
        cbar.set_label(f"Summed anticorrelation", rotation=270, labelpad=30)

        if self.options["invert_x"]:
            ax.invert_xaxis()
        if self.options["invert_y"]:
            ax.invert_yaxis()


if __name__ == '__main__':
    sample_idx = 1

    meas_dir_sub = data_dir / "Uncoated" / sample_names[sample_idx]
    sub_image = Image(data_path=meas_dir_sub)

    # meas_dir = data_dir / "Edge" / sample_names[sample_idx]
    # meas_dir = data_dir / "Edge_4pp2_flipped" / (sample_names[sample_idx] + "_accident")
    meas_dir = data_dir / "Edge_4pp2_flipped" / sample_names[sample_idx]
    # meas_dir = data_dir / "Edge_4pp2" / sample_names[sample_idx]
    # options = {"excluded_areas": [[3, 13, -10, 30], [33, 35, -10, 30]], "cbar_min": 1.0e6, "cbar_max": 6.5e6}
    options = {"excluded_areas": [[-10, 55, 12, 30],
                                  [-10, -6, -10, 30],
                                  [37, 55, -10, 30], ]}
    options = {"excluded_areas": [[33, 60, -4, 30],  # s1, s2, s3
                                  [3, 11, -4, 30],
                                  # [36, 55, -4, 30],
                                  ]}
    """
    options = {"excluded_areas": [  # s4
        [-13, 60, 13, 30],
        [-10, -5, -5, 30],
        [37, 60, -5, 30]
    ]}
    """  # 1.60*1e4, 1.74*1e4
    """
    options = {"cbar_min": 1.47e5, "cbar_max": 2.9e5, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": True}  # s4 (idx 3)
    """
    options = {"cbar_min": 1.5e4, "cbar_max": 1.85e4, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": False}  # s2 (idx 1) unflipped
    options = {"cbar_min": 0, "cbar_max": np.inf, "log_scale": False, "color_map": "viridis",
               "invert_x": True, "invert_y": False}  # s2 (idx 1) unflipped

    options = {"cbar_min": 1.4e4, "cbar_max": 1.70e4, "log_scale": False, "color_map": "viridis",
               "invert_x": False, "invert_y": False}  # s2 (idx 1) flipped

    film_image = Image(meas_dir, sub_image, sample_idx, options)
    # s1, s2, s3 = [-10, 50, -3, 27]
    # film_image.plot_cond_vs_d()
    # film_image.plot_image(img_extent=[-10, 50, -3, 27], quantity="loss", selected_freq=1.200)
    # sub_image.plot_image(img_extent=[-10, 50, -3, 27], quantity="p2p")
    # film_image.plot_image(quantity="p2p")
    # film_image.plot_image(quantity="Conductivity", selected_freq=1.200)
    # film_image.thz_vs_4pp(row_idx=2, p0=(40, 10.5))  # s2
    film_image.thz_vs_4pp(row_idx=2, p0=(33.6, 10.5))  # s2 flipped
    # film_image.thz_vs_4pp(row_idx=2, p0=(43, 4))  # s2 from corr img.
    # film_image.thz_vs_4pp(row_idx=3, p0=(37, 6)) # s4
    # film_image.correlation_image(row_idx=2, p0=(33.6, 10.5))  # s2 flipped
    # film_image.correlation_image(row_idx=3, p0=(37, 6))  # s4
    # film_image.plot_image(img_extent=[-10, 50, -3, 27], quantity="Reference phase", selected_freq=1.200)

    # sub_image.system_stability(selected_freq_=1.200)
    # film_image.system_stability(selected_freq_=1.200)

    # s4 = [18, 51, 0, 20]
    # film_image.plot_image(img_extent=[18, 51, 0, 20], quantity="p2p", selected_freq=1.200)
    # film_image.plot_image(img_extent=[18, 51, 0, 20], quantity="Conductivity", selected_freq=0.600)
    # film_image.plot_image(img_extent=[-5, 36, 0, 12], quantity="Conductivity", selected_freq=1.200)
    # film_image.plot_image(img_extent=[18, 51, 0, 20], quantity="power", selected_freq=(1.150, 1.250))

    # stability_dir = data_dir / "Stability" / "2023-03-20"

    # stability_image = Image(stability_dir)
    # stability_image.system_stability(selected_freq_=1.200)

    for fig_label in plt.get_figlabels():
        if "Sample" in fig_label:
            continue
        plt.figure(fig_label)
        plt.legend()

    plt.show()
