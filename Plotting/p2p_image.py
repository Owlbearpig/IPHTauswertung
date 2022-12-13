from imports import *


def p2p_image(refs, sams, point_value="rel_p2p"):
    x_positions = [meas.position[0] for meas in sams]
    y_positions = [meas.position[1] for meas in sams]

    min_x, min_y = min(x_positions), min(y_positions)
    max_x, max_y = max(x_positions), max(y_positions)

    bounds = [[min_x, max_x], [min_y, max_y]]

    unique_x, unique_y = sorted(list(set(x_positions))), sorted(list(set(y_positions)))

    rez_x, rez_y = len(unique_x), len(unique_y)
    grd_x = np.linspace(bounds[0][0], bounds[0][1], rez_x)
    grd_y = np.linspace(bounds[1][0], bounds[1][1], rez_y)

    grid_vals = np.zeros((rez_x, rez_y))
    refs = sorted(refs, key=lambda ref: ref.meas_time)
    sams = sorted(sams, key=lambda sam: sam.meas_time)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cbar_label = ""

    for i in range(len(sams)):
        matched_ref_idx = np.argmin([np.abs(sams[i].meas_time - ref_i.meas_time) for ref_i in refs])
        matched_ref = refs[matched_ref_idx]

        sam_fd, ref_fd = sams[i].get_data_fd(), matched_ref.get_data_fd()
        sam_td_data, ref_td_data = sams[i].get_data_td(), matched_ref.get_data_td()
        edge_val = 0.785
        # relative peak to peak value
        if "rel_p2p" in point_value:
            ax.set_title("Relative peak to peak value")
            cbar_label = r"p2p($y_{sam}$) / p2p($y_{ref}$)"

            p2p_val_sam = np.abs(np.max(sam_td_data[:, 1]) - np.min(sam_td_data[:, 1]))
            p2p_val_ref = np.abs(np.max(ref_td_data[:, 1]) - np.min(ref_td_data[:, 1]))
            val = p2p_val_sam / p2p_val_ref
        elif "integrated_intensity" in point_value:
            ax.set_title("Integrated spectra over 0.5 THz - 1.0 THz")
            cbar_label = "$\sum |FFT(y_{sam})| $ / $\sum |FFT(y_{ref})|$"
            edge_val = 0.80

            val = np.sum(np.abs(sam_fd[50:100])) / np.sum(np.abs(ref_fd[50:100]))
        elif "tof" in point_value:  # time of flight
            ax.set_title("ToF (pp_sam - pp_ref) image")
            cbar_label = "$\Delta$t (ps)"
            edge_val = 0
            argmax_sam = np.argmax(np.abs(sam_td_data[:, 1]))
            argmax_ref = np.argmax(np.abs(ref_td_data[:, 1]))
            val = sam_td_data[argmax_sam, 0] - sam_td_data[argmax_ref, 0]
        elif "pulse_position" in point_value:
            cbar_label = ""

            argmax_sam = np.argmax(np.abs(sam_td_data[:, 1]))
            val = sam_td_data[argmax_sam, 0]
        elif "avg_ref_p2p_value" in point_value:
            cbar_label = ""

            if sams[i].meas_time < matched_ref.meas_time:
                shift = -1
            else:
                shift = 1
            if matched_ref_idx != (len(refs) - 1):
                matched_ref_next = refs[matched_ref_idx + shift]
            else:
                matched_ref_next = refs[matched_ref_idx]

            ref_next_td_data = matched_ref_next.get_data_td()
            avg_ref_td = ref_td_data.copy()
            avg_ref_td[:, 1] = (ref_next_td_data[:, 1] + ref_td_data[:, 1]) / 2
            p2p_val_sam = np.abs(np.max(sam_td_data[:, 1]) - np.min(sam_td_data[:, 1]))
            p2p_avg_ref = np.abs(np.max(avg_ref_td[:, 1]) - np.min(avg_ref_td[:, 1]))
            val = p2p_val_sam / p2p_avg_ref
        else:
            val = 0

        x_pos, y_pos = sams[i].position
        #y_pos = abs(y_pos - 20) # rotate y around y=10
        if x_pos > 55:
            val = edge_val
        grid_vals[unique_x.index(x_pos), unique_y.index(y_pos)] = val

    fig.subplots_adjust(left=0.2)
    extent = [grd_x[0], grd_x[-1], grd_y[0], grd_y[-1]] # correct
    #extent = [grd_x[0], grd_x[-1], grd_y[-1], grd_y[0]]  # flipped y axis
    aspect = ((bounds[0][1] - bounds[0][0]) / rez_x) / ((bounds[1][1] - bounds[1][0]) / rez_y)

    img = ax.imshow(grid_vals[:, :].transpose((1, 0)), vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                    origin="lower",
                    cmap=plt.get_cmap("jet"),
                    extent=extent,
                    aspect=aspect)

    ax.set_xlabel("Horizontal stage pos. x (mm)")
    ax.set_ylabel("Vertical stage pos. y (mm)")

    cbar = fig.colorbar(img)
    cbar.set_label(cbar_label, rotation=270, labelpad=30)
