from imports import *


def p2p_image(refs, sams):
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

    for i in range(len(sams)):
        matched_ref_idx = np.argmin([np.abs(sams[i].meas_time - ref_i.meas_time) for ref_i in refs])

        matched_ref = refs[matched_ref_idx]
        if sams[i].meas_time < matched_ref.meas_time:
            shift = -1
        else:
            shift = 1
        if matched_ref_idx != (len(refs) - 1):
            matched_ref_next = refs[matched_ref_idx + shift]
        else:
            matched_ref_next = refs[matched_ref_idx]
        print(sams[i].meas_time)
        print(matched_ref.meas_time, matched_ref_next.meas_time)

        sam_td_data, ref_td_data = sams[i].get_data_td(), matched_ref.get_data_td()
        ref_next_td_data = matched_ref_next.get_data_td()
        avg_ref_td = ref_td_data.copy()
        avg_ref_td[:, 1] = (ref_next_td_data[:, 1] + ref_td_data[:, 1]) / 2
        p2p_val_sam = np.abs(np.max(sam_td_data[:, 1]) - np.min(sam_td_data[:, 1]))
        #p2p_val_ref = np.abs(np.max(ref_td_data[:, 1]) - np.min(ref_td_data[:, 1]))
        #p2p_val_ref_next = np.abs(np.max(ref_next_td_data[:, 1]) - np.min(ref_next_td_data[:, 1]))
        p2p_avg_ref = np.abs(np.max(avg_ref_td[:, 1]) - np.min(avg_ref_td[:, 1]))
        val = p2p_val_sam / p2p_avg_ref

        #argmax_sam = np.argmax(np.abs(sam_td_data[:, 1]))
        #argmax_ref = np.argmax(np.abs(ref_td_data[:, 1]))
        #val = sam_td_data[argmax_sam, 0] - sam_td_data[argmax_ref, 0]
        #val = sam_td_data[argmax_sam, 0]

        x_pos, y_pos = sams[i].position
        grid_vals[unique_x.index(x_pos), unique_y.index(y_pos)] = val

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_title("ToF (pp_sam - pp_ref) image")
    ax.set_title("P2p image")
    fig.subplots_adjust(left=0.2)
    extent = [grd_x[0], grd_x[-1], grd_y[0], grd_y[-1]]
    aspect = ((bounds[0][1] - bounds[0][0]) / rez_x) / ((bounds[1][1] - bounds[1][0]) / rez_y)

    img = ax.imshow(grid_vals[:, :].transpose((1, 0)), vmin=np.min(grid_vals), vmax=np.max(grid_vals),
                    origin="lower",
                    cmap=plt.get_cmap("jet"),
                    extent=extent,
                    aspect=aspect)

    ax.set_xlabel("Horizontal stage pos. x (mm)")
    ax.set_ylabel("Vertical stage pos. y (mm)")

    cbar = fig.colorbar(img)
    cbar.set_label("p2p_sam / p2p_ref", rotation=270, labelpad=20)

