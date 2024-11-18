import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os


# plt.style.use('seaborn')
# plt.style.use('tex')

def set_size(width, inch=True, fraction=1, num_figures=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    if inch:
        fig_width_pt = width * 72.27
    else:
        fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * num_figures

    fig_dim = (fig_width_in, fig_height_in)

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 8,
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 3.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.figsize": fig_dim,
        'lines.linewidth': 0.4,
        'figure.facecolor': "white"
    }

    plt.rcParams.update(tex_fonts)


set_size(240, fraction=1)

def plot_stairs(x, y, label, xlabel, ylabel, title, file_name):
    plt.figure()
    plt.stairs(y, x, baseline=0, fill=True, orientation='vertical', color='tab:blue', alpha=0.5, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 1)
    plt.title(title)

    plt.grid(True, which='major', color='grey', alpha=0.2, linewidth=0.3)
    plt.grid(True, which='minor', color='grey', linestyle='--', alpha=0.1, linewidth=0.1)
    # plt.minorticks_on()
    # plt.tick_params(which='major', axis="y", direction="in", width=0.5, color='grey')
    # plt.tick_params(which='minor', axis="y", direction="in", width=0.3, color='grey')
    # plt.tick_params(which='major', axis="x", direction="in", width=0.5, color='grey')
    # plt.tick_params(which='minor', axis="x", direction="in", width=0.3, color='grey')
    ax = plt.gca()

    # Remove the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('grey')
    ax.spines['left'].set_linewidth(0.5)

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.01)


def plot(x, y, hline=None, title="", xlabel="", ylabel="", legend=None, save_path=None, new_figure=True, label="",
         mark_max=False, semilogy=False, semilogx=False, target_accuracy=None, ylim=None, mark_height=None, 
         xlim = None, legend_loc="upper right", **kwargs):
    if semilogx and semilogy:
        plot_func = plt.loglog
    elif semilogx:
        plot_func = plt.semilogx
    elif semilogy:
        plot_func = plt.semilogy
    else:
        plot_func = plt.plot
    if new_figure:
        plt.figure()
    if hline is not None:
        plot_func(x, y, x, len(x) * [hline], **kwargs)
    else:
        line, = plot_func(x, y, label=label, **kwargs)
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([0, max(x)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='major', color='grey', alpha=0.2, linewidth=0.3)
    plt.grid(True, which='minor', color='grey', linestyle='--', alpha=0.1, linewidth=0.1)
    plt.minorticks_on()
    plt.tick_params(which='major', axis="y", direction="in", width=0.5, color='grey')
    plt.tick_params(which='minor', axis="y", direction="in", width=0.3, color='grey')
    plt.tick_params(which='major', axis="x", direction="in", width=0.5, color='grey')
    plt.tick_params(which='minor', axis="x", direction="in", width=0.3, color='grey')
    ax = plt.gca()

    # Remove the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('grey')
    ax.spines['left'].set_linewidth(0.5)
    if mark_max:
        if target_accuracy is None:
            max_idx = np.where(y >= np.max(y))[0][0]
        else:
            max_idx = np.where(y >= target_accuracy)[0][0]
        plot_func([x[max_idx], x[max_idx]], [100 if mark_height is None else mark_height, 0], "--", c=line.get_color(), **kwargs)
        plt.text(x[max_idx], 103 if mark_height is None else mark_height, f"{int(x[max_idx])}", size=5, rotation=60)
        plt.ylim([0, 115] if ylim is None else ylim)
    if legend is not None:
        plt.legend(legend, frameon=True, loc=legend_loc)
    else:
        plt.legend(frameon=True, loc=legend_loc)
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')


def plot_loss_from_csv(file_dir, legend, output_dir, title):
    set_size(3.25, inch=True, fraction=1, num_figures=1)
    if not os.path.exists("figures"):
        os.makedirs("figures")
    if "phase1" in file_dir:
        phase = 1
    elif "phase2" in file_dir:
        phase = 2
    else:
        raise ValueError("Phase not found in file_dir")
    figure_path = f"figures/{output_dir}"
    data = pd.read_csv(f"results/{file_dir}/phase{phase}_log_metrics.csv")
    loss = data["step_loss"].values
    iteration = data["step"].values
    if iteration[0] != 1:
        iteration = iteration - iteration[0] + 1
    plt.figure
    plot(iteration, loss, new_figure=False, xlabel="Iterations", 
         ylabel="Training Loss", label=legend, save_path=figure_path, title=title)
    

def plot_cosine_similarity(file_name):
    set_size(3.25, inch=True, fraction=1, num_figures=1)
    if not os.path.exists("figures"):
        os.makedirs("figures")
    save_path = "figures/lora_convergence.pdf"
    data = pd.read_csv(file_name)
    iterations = data["Iteration"].values
    iterations -= iterations[0]
    plt.figure()
    for layer in ["Query", "Key", "Value", "Projection", "Upsample", "Downsample"]:
        relevant_columns = []
        for column in data.columns:
            if layer in column and "lora" in column:
                relevant_columns.append(column)
        similarity = data[relevant_columns].values.mean(axis=1)
        plot(iterations, similarity, new_figure=False, xlabel="Iterations",
              ylabel="Cosine Similarity", label=layer, save_path=save_path, 
              title="Low-Rank Adapter Similarity with Converged Weight", legend_loc="lower right")
        

    # for i in range(24):
    #     relevant_columns = []
    #     for column in data.columns:
    #         if str(i) in column and "lora" in column:
    #             relevant_columns.append(column)
    #     similarity = data[relevant_columns].values
    #     print(relevant_columns, similarity.shape)
            

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--lora_convergence", action="store_true")
    arg_parser.add_argument("--file_list", type=str, required=True)
    arg_parser.add_argument("--legend_list", type=str)
    arg_parser.add_argument("--output_dir", type=str)
    arg_parser.add_argument("--title", type=str)
    args = arg_parser.parse_args()

    if args.lora_convergence:
        plot_cosine_similarity(args.file_list)
    else:
        file_list = args.file_list.split(",")
        legend_list = args.legend_list.split(",")

        if len(file_list) != len(legend_list):
            raise ValueError("Number of files and legends must be the same")

        title = args.title.replace("_", " ")

        for file, legend in zip(file_list, legend_list):
            legend = legend.replace("_", " ")
            plot_loss_from_csv(file, legend, args.output_dir, title)