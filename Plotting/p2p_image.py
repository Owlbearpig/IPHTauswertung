from imports import *


def p2p_image(measurements):
    x_positions = [meas.position[0] for meas in measurements]
    y_positions = [meas.position[1] for meas in measurements]

    min_x, min_y = min(x_positions), min(y_positions)
    max_x, max_y = max(x_positions), max(y_positions)

    bounds = [[min_x, max_x], [min_y, max_y]]

    unique_x, unique_y = sorted(list(set(x_positions))), sorted(list(set(y_positions)))

    rez_x, rez_y = len(unique_x), len(unique_y)
    grd_x = np.linspace(bounds[0][0], bounds[0][1], rez_x)
    grd_y = np.linspace(bounds[1][0], bounds[1][1], rez_y)

    grid_vals = np.zeros((rez_x, rez_y))

    for measurement in measurements:
        td_data = measurement.get_data_td()
        val = np.abs(np.max(td_data[:, 1])-np.min(td_data[:,1]))
        x_pos, y_pos = measurement.position
        grid_vals[unique_x.index(x_pos), unique_y.index(y_pos)] = val

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("P2p image")
    fig.subplots_adjust(left=0.2)
    extent = [grd_x[0], grd_x[-1], grd_y[0], grd_y[-1]]
    aspect = ((bounds[0][1] - bounds[0][0]) / rez_x) / ((bounds[1][1] - bounds[1][0]) / rez_y)

    img = ax.imshow(grid_vals[:, :].transpose((1, 0)), vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                    origin="lower",
                    cmap=plt.get_cmap("jet"),
                    extent=extent,
                    aspect=aspect)

    ax.set_xlabel("horizontal position, x (mm)")
    ax.set_ylabel("vertical position, y (mm)")

    cbar = fig.colorbar(img)
    cbar.set_label("p2p value (a.u.)", rotation=270, labelpad=20)

