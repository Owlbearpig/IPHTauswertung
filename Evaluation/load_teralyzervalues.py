import numpy as np
import matplotlib.pyplot as plt
from consts import *
import pandas as pd

teralyzer_dir = (data_dir / "Uncoated/10x10cm_sqrd_s1/TeraLyzer_s1").glob("**/*.csv")


def teralyzer_read_point(x, y):
    for file in teralyzer_dir:
        file_s = str(file)
        parts = file_s.split("_")
        x_s, y_s = parts[-3:-1]
        x_coord, y_coord = float(x_s.replace("x", "")), float(y_s.replace("y", ""))
        if np.isclose(x_coord-x, 0) and np.isclose(y_coord-y, 0):
            df = pd.read_csv(file)
            data = df.values
            n = np.array([data[:, 0]/THz, data[:, 1] + 1j * data[:, 2]], dtype=complex).T

            return n

    print(f"File with coordinates x={x}, y={y}, not found in {teralyzer_dir}")


if __name__ == '__main__':
    n_8_12 = teralyzer_read_point(8, 12)

    plt.plot(n_8_12[:, 0], n_8_12[:, 1].real, label="Real part")
    plt.plot(n_8_12[:, 0], n_8_12[:, 1].imag, label="Imaginary part")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Complex refractive index")
    plt.legend()
    plt.show()



