import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from typing import Dict, Optional, List


def plot_energy(energy_data: Dict[float, float],
                inference_data: Dict[float, float],
                min_energy: float,
                max_energy: float,
                output_folder: Optional[RichPath]):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()

        # Get style colors
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Collect the energy values
        times = [t for t, _ in energy_data.items()]
        energy_values = [e for _, e in energy_data.items()]

        # Plot the energy values
        ax.plot(times, energy_values, label='Energy', color=colors[0])
        ax.set_title('Energy Levels and Inference Rate')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Energy (J)', color=colors[0])
        ax.tick_params(axis='y', labelcolor=colors[0])

        # Plot min/max energy values
        ax.axhline(y=min_energy, xmin=0, xmax=max(times), color='black', linestyle='--')
        ax.axhline(y=max_energy, xmin=0, xmax=max(times), color='black', linestyle='--')

        # Collect the inference values
        times = [t for t, _ in inference_data.items()]
        inference_rate = [n for _, n in inference_data.items()]
        
        # Plot the inference values
        ax2 = ax.twinx()
        ax2.plot(times, inference_rate, label='Inference Rate', color=colors[1])
        ax2.set_ylabel('Inference Rate (sec / op)', color=colors[1])
        ax2.tick_params(axis='y', labelcolor=colors[1])
        ax2.grid(False)

        fig.tight_layout()

        if output_folder is not None:
            output_file = output_folder.join('energy_plot.pdf')
            plt.savefig(output_file.path)
        else:
            plt.show()

    plt.close()

def plot_levels(selected_levels: List[int], max_num_levels: int,  output_folder: RichPath):

    with plt.style.context('ggplot'):
        # Get style colors
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots()

        for level in range(1, max_num_levels + 1):
            xs = [i for i, ell in enumerate(selected_levels) if ell == level]
            ys = [level for _ in range(len(xs))]
            ax.scatter(xs, ys, color=colors[level - 1], label=f'Level {level}')

        ax.set_title('Selected Output Levels per Iteration')
        ax.set_ylabel('Selected Level')
        ax.set_xlabel('Iteration')

        if output_folder is not None:
            output_file = output_folder.join('selected_levels.pdf')
            plt.savefig(output_file.path)
        else:
            plt.show()

    plt.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--min-energy', type=float, required=True)
    parser.add_argument('--max-energy', type=float, required=True)
    parser.add_argument('--output-folder', type=str)
    args = parser.parse_args()

    data_folder = RichPath.create(args.data_folder)
    assert data_folder.exists(), f'The folder {data_folder} does not exist!'

    energy_data = data_folder.join('energy_results.pkl.gz').read_by_file_suffix()
    inference_data = data_folder.join('inference_results.pkl.gz').read_by_file_suffix()
    output_folder = RichPath.create(args.output_folder) if args.output_folder is not None else args.output_folder

    plot_energy(enery_data=energy_data,
                inference_data=inference_data,
                min_energy=args.min_energy,
                max_energy=args.max_energy,
                output_folder=output_folder)
