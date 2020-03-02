import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from typing import Dict, Optional, List, Union


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

def plot_levels(selected_levels: List[int], max_num_levels: int,  output_folder: Optional[RichPath]):

    with plt.style.context('ggplot'):
        # Get style colors
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots()

        for level in range(1, max_num_levels + 1):
            xs = [i for i, ell in enumerate(selected_levels) if ell == level]
            ys = [level for _ in range(len(xs))]
            ax.scatter(xs, ys, color=colors[level - 1], label=f'Level {level}')

        # Set titles
        ax.set_title('Selected Output Levels per Iteration')
        ax.set_ylabel('Selected Level')
        ax.set_xlabel('Iteration')

        fig.tight_layout()

        if output_folder is not None:
            output_file = output_folder.join('selected_levels.pdf')
            plt.savefig(output_file.path)
        else:
            plt.show()

    plt.close()


def plot_errors(errors_dicts: Union[List[Dict[float, float]], Dict[float, float]], labels: Union[List[str], str], output_folder: Optional[RichPath]):
    if not isinstance(errors_dicts, list):
        errors_dicts = [errors_dicts]
        labels = [labels]

    with plt.style.context('ggplot'):

        fig, ax = plt.subplots()

        for label, errors_dict in zip(labels, errors_dicts):
            times: List[float] = []
            mse: List[float] = []
            
            error_sum, num_trials = 0.0, 0.0
            for time in sorted(errors_dict.keys()):
                error_sum += errors_dict[time]
                num_trials += 1

                # Appends to lists
                times.append(time)
                mse.append(error_sum / num_trials)

            ax.plot(times, mse, label=label)

        # Set labels
        ax.set_ylabel('Mean Squared Error (MSE)')
        ax.set_xlabel('Time')
        ax.set_title('Mean Squared Error over Time')
        fig.tight_layout()

        if output_folder is not None:
            output_file = output_folder.join('mse_time.pdf')
            plt.savefig(output_file.path)
        else:
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folders', type=str, nargs='+')
    parser.add_argument('--labels', type=str, nargs='?')
    parser.add_argument('--min-energy', type=float, required=True)
    parser.add_argument('--max-energy', type=float, required=True)
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--mode', choices=['energy', 'error'])
    args = parser.parse_args()

    data_folders: List[RichPath] = []
    for data_folder in args.data_folders:
        path = RichPath.create(data_folder)
        assert path.exists(), f'The folder {path} does not exist!'
        data_folders.append(path)

    output_folder = RichPath.create(args.output_folder) if args.output_folder is not None else args.output_folder

    if (mode == 'energy'):
        energy_data = data_folders[0].join('energy_results.pkl.gz').read_by_file_suffix()
        inference_data = data_folders[0].join('inference_results.pkl.gz').read_by_file_suffix()
        plot_energy(enery_data=energy_data,
                    inference_data=inference_data,
                    min_energy=args.min_energy,
                    max_energy=args.max_energy,
                    output_folder=output_folder)
    elif (mode == 'error'):
        errors_dicts = [df.join('error_results.pkl.gz').read_by_file_suffix() for df in data_folders]
        assert len(args.labels) == len(errors_dicts), f'Got {len(args.labels)} labels but {len(errors_dicts)} error dictionaries'
        plot_errors(errors_dcts, args.labels, output_folder=output_folder)
