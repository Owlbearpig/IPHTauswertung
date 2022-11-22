from imports import *
from functions import calc_absorption
from Optimization.cost_function import Cost
from functools import partial
from Model.transmission_approximation import ri_approx


def main():
    d_list = [inf, 500, inf]
    keywords = ["01 GaAs Wafer 25", "2022_02_14"]

    new_cost = Cost(d_list, keywords, sam_idx=8, simulated_sample=False)
    freqs = new_cost.freqs
    freq = 1.300
    selected_freq_idx = np.argmin(np.abs(freqs - freq))
    print(f"Selected frequency: {freqs[selected_freq_idx]} (THz), idx: {selected_freq_idx}")

    cost_func = partial(new_cost.cost, selected_freq_idx)

    bounds = [[3.4, 3.9], [0.002, 0.020]]

    rez_x, rez_y = 350, 350
    grd_x = np.linspace(bounds[0][0], bounds[0][1], rez_x)
    grd_y = np.linspace(bounds[1][0], bounds[1][1], rez_y)

    grid_vals = np.zeros((rez_x, rez_y))

    for i in range(rez_x):
        for j in range(rez_y):
            p = array([grd_x[i], grd_y[j]])
            grid_vals[i, j] = cost_func(p)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Residual sum plot")
    fig.subplots_adjust(left=0.2)
    extent = [grd_x[0], grd_x[-1], grd_y[0], grd_y[-1]]
    aspect = ((bounds[0][1] - bounds[0][0]) / rez_x) / ((bounds[1][1] - bounds[1][0]) / rez_y)
    img = ax.imshow(grid_vals[:, :].transpose((1, 0)), vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                    origin="lower",
                    cmap=plt.get_cmap("jet"),
                    extent=extent,
                    aspect=aspect)

    ax.set_xlabel("n")
    ax.set_ylabel("k")

    g_min_idx = np.argmin(grid_vals)
    min_x, min_y = np.unravel_index(g_min_idx, grid_vals.shape)

    p_found = [grd_x[min_x], grd_y[min_y]]
    print(f"Found: {p_found}, fx={cost_func(p_found)}")

    p_goal = [new_cost.n_approx[selected_freq_idx, 1].real, new_cost.n_approx[selected_freq_idx, 1].imag]
    print(f"Goal: {p_goal}, fx={cost_func(p_goal)}")

    cbar = fig.colorbar(img)
    cbar.set_label("loss", rotation=270, labelpad=20)


if __name__ == '__main__':
    main()
    plt.show()
