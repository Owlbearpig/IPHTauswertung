import matplotlib as mpl
from consts import cur_os, Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc


# print(rcParams.keys())

# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

def fmt(x, val):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    # return r'${} \times 10^{{{}}}$'.format(a, b)
    return rf'{a}E+{b:02}'


def mpl_style_params():
    rcParams = mpl.rcParams
    # rcParams['lines.linestyle'] = '--'
    # rcParams['legend.fontsize'] = 'large' #'x-large'
    rcParams['legend.shadow'] = False
    # rcParams['lines.marker'] = 'o'
    rcParams['lines.markersize'] = 4
    rcParams['lines.linewidth'] = 3.5  # 2
    rcParams['ytick.major.width'] = 2.5
    rcParams['xtick.major.width'] = 2.5
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    # rcParams['axes.grid'] = True
    rcParams['figure.autolayout'] = False
    rcParams['savefig.format'] = 'jpg'
    rcParams.update({'font.size': 24})

    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['DejaVu Serif']

    #"""
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX to write all text
        "font.family": "serif",  # Use serif fonts
        "font.serif": ["Computer Modern"],  # Ensure it matches LaTeX default font
        # "text.latex.preamble": r"\\usepackage{amsmath}"  # Add more packages as needed
    })
    # """
    if 'posix' in cur_os:
        result_dir = Path(r"/home/alex/MEGA/AG/Projects/Conductivity/IPHT/Publication/Figures")
    else:
        # result_dir = Path(r"E:\Mega\AG\Projects\THz Conductivity\IPHT\Results")
        result_dir = Path(r"E:\Mega\AG\Projects\Conductivity\IPHT\Publication\Figures")
    rcParams["savefig.directory"] = result_dir

    return rcParams


if __name__ == '__main__':
    mpl_style_params()

    from matplotlib.pyplot import subplots, xlabel, ylabel, grid, show

    fig, ay = subplots()

    # Using the specialized math font elsewhere, plus a different font
    xlabel(r"Film conductivity", fontsize=27)
    # No math formatting, for comparison
    ylabel(r'Italic and just Arial and not-math-font', fontsize=18)
    grid()

    show()

    exit()
