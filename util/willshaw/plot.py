import matplotlib.pyplot as plt


def plot_multiple_line_charts(x_l, y_l_l, label_l, title, xlabel, ylabel):
    for y_l, label in zip(y_l_l, label_l):
        plt.plot(x_l, y_l, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim((0, 0.1))
    plt.legend()
    plt.savefig(f"img/willshaw/{title}_multiple_runs.png")
