from consts import data_dir
from Measurements.image import Image
import matplotlib.pyplot as plt
from functions import f_axis_idx_map, export_spectral_array

sub_eval_pt = (30, 10)
film_eval_pt = (5, -2)

sample_idx = 3

meas_dir_sub = data_dir / "Uncoated" / "s4"
sub_image = Image(data_path=meas_dir_sub, options={"load_mpl_style": False})

absorb = sub_image.get_absorbance(sub_eval_pt)

export_spectral_array(absorb, "sub_absorbance", (0.25, 3.0))

meas_dir_film = data_dir / "s4_new_area" / "Image0"

film_image = Image(meas_dir_film, sub_image, sample_idx)
film_image.get_absorbance(film_eval_pt, en_plot=True)

plt.show()
