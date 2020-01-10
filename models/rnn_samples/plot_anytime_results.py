import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from typing import Dict


def plot_energy(energy_data: Dict[float, float], min_energy: float, max_energy: float):
    with plt.style.context('ggplot'):
        times = [t for t, _ in energy_data.items()]
        energy_values = [e for _, e in energy_data.items()]

        fig, ax = plt.subplots()
        ax.plot(times, energy_values)

        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')

        plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--min-energy', type=float, required=True)
    parser.add_argument('--max-energy', type=float, required=True)
    args = parser.parse_args()

    data_folder = RichPath.create(args.data_folder)
    assert data_folder.exists(), f'The folder {data_folder} does not exist!'

    energy_data = data_folder.join('energy_results.pkl.gz').read_by_file_suffix()

    plot_energy(enery_data=energy_data,
                min_energy=args.min_energy,
                max_energy=args.max_energy)
