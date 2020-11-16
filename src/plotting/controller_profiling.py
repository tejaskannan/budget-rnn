import matplotlib.pyplot as plt
from typing import Optional


energy_series = ['LEA, RAM', 'LEA, FRAM+DMA']
energy_per_dim = {
    10: [12.917, 12.951],
    20: [13.076, 13.096],
    30: [13.196, 13.231],
    40: [13.23, 13.246]
}


cycles_series = ['LEA, RAM', 'LEA, FRAM+DMA', 'No LEA, RAM', 'LEA, FRAM']
cycles_per_dim = {
    10: [3477, 4179, 3588, 5795],
    20: [7027, 8659, 12428, 15425],
    30: [11077, 14039, 26668, 29355],
    40: [15627, 20319, 46330, 47585]
}


def plot_energy(output_file: Optional[str]):
    with plt.style.context('ggplot'):

        xs: List[float] = []
        ys: List[List[float]] = [[] for _ in energy_series]

        for dim, energy_list in energy_per_dim.items():
            xs.append(dim)

            for series_idx, energy in enumerate(energy_list):
                ys[series_idx].append(energy)

        fig, ax = plt.subplots()

        ax.set_title('Energy for 10 seconds of Matrix-Vector Products')
        ax.set_xlabel('Matrix Dimensions')
        ax.set_ylabel('Energy (mJ)')

        for energy, label in zip(ys, energy_series):
            ax.plot(xs, energy, label=label, marker='o', markersize=3)

            for d, e in zip(xs, energy):
                sign = -1 if label == 'LEA, RAM' else 1
                xshift = 1 if label == 'LEA, RAM' else -2.5

                ax.annotate('{0:.3f}'.format(e), xy=(d, e), xytext=(d + xshift, e + sign * 0.01))

        ax.legend()

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


def plot_cycles(output_file: Optional[str]):
    with plt.style.context('ggplot'):

        xs: List[float] = []
        ys: List[List[float]] = [[] for _ in cycles_series]

        for dim, cycles_list in cycles_per_dim.items():
            xs.append(dim)

            for series_idx, cycles in enumerate(cycles_list):
                ys[series_idx].append(cycles)

        fig, ax = plt.subplots()

        ax.set_title('Cycles for Matrix-Vector Products')
        ax.set_xlabel('Matrix Dimensions')
        ax.set_ylabel('CPU Cycles')

        for cycles, label in zip(ys, cycles_series):
            ax.plot(xs, cycles, label=label, marker='o', markersize=3)

        ax.legend()

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    plot_cycles(output_file='mcu_experiments/cycles.png')
    plot_energy(output_file='mcu_experiments/energy.png')
