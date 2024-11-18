import torch
import matplotlib.pyplot as plt
import os
import numpy as np


def set_size(width, fraction=1):
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
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


fig_dim = set_size(450, fraction=1)

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": False,
    "font.family": "serif",
    "axes.titlesize": 8,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 8,
    "font.size": 2,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 2,
    "xtick.labelsize": 2,
    "ytick.labelsize": 2,
    "figure.figsize": fig_dim,
    'lines.linewidth': 0.4,
    'figure.facecolor': "white"
}


plt.rcParams.update(tex_fonts)


def pie_plot(data_dict, title="", save_path=None):
    val_list = []
    label_list = []

    for label, val in data_dict.items():
        if label == "train_time":
            continue
        val_list.append(val)
        label_list.append(label)

    val_list = np.array(val_list)
    val_list, label_list = zip(*sorted(zip(val_list, label_list), reverse=True))
    val_list = np.array(val_list)
    percentage = 100. * (val_list / val_list.sum())
    patches, texts = plt.pie(val_list, startangle=90)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(label_list, percentage)]
    plt.legend(patches, labels, loc="center left", bbox_to_anchor=(-0.35, 0.5), fontsize=8, frameon=True)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')


class Timer():
    def __init__(self, measure=False):
        self.time = {}
        self.measure = measure
        
    def __call__(self, name, func, *args, **kwargs):
        if self.measure:
            if name not in self.time:
                self.time[name] = 0.0
                self.__setattr__(name + "start", torch.cuda.Event(enable_timing=True))
                self.__setattr__(name + "end", torch.cuda.Event(enable_timing=True))
            start = self.__getattribute__(name + "start")
            end = self.__getattribute__(name + "end")
            start.record()
        output = func(*args, **kwargs)
        if self.measure:
            end.record()
            torch.cuda.synchronize()
            self.time[name] += start.elapsed_time(end)
        return output
    
    def zero_time(self):
        if self.measure:
            for key in self.time:
                self.time[key] = 0.0
    
    def save(self, path, title="Timing"):
        self.time = self.get_timer_dict()
        if not os.path.exists(path):
            os.makedir(path)
        torch.save(self.time, path + f"/timing_{torch.cuda.get_device_name()}.time")
        pie_plot(self.time, title=title, save_path=path + f"/timing_{torch.cuda.get_device_name()}.pdf")
        

    def get_timer_dict(self):
        if hasattr(self, "train_time"):
            self.time["train_time"] = self.train_time
        return self.time

    def combine_timing(self, optimizer):
        if self.measure:
            if hasattr(optimizer, "timer"):
                if optimizer.timer.measure == False:
                    return
                del self.time["optimizer"]
                for key in optimizer.timer.time:
                    self.time[key] = optimizer.timer.time[key]