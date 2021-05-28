import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List


def plot(x: List[float], ys: List[List[float]], series_names: List[str], title: str, x_label: str, y_label: str, output_file_name: str):
    if len(ys) != len(series_names):
        raise ValueError(f'The number of y-series {len(ys)} must be the same as the number of series names {len(series_names)}.')
    
    with plt.style.context('ggplot'):

        for name, y_vals in zip(series_names, ys):
            plt.plot(x, y_vals, marker='o', label=name)

        plt.xticks(ticks=x, labels=x)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.savefig(output_file_name + '.pdf')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-file-name', type=str, required=True)
    args = parser.parse_args()

    x = [0.2, 0.4, 0.6, 0.8, 1.0]
    ys = [[0.390428441209875, 0.363065372987015, 0.334097436352971, 0.301025186902277, 0.271514891395549], \
          [0.426481617416093, 0.433547572123391, 0.486339870122521, 0.390288696173765, 0.266194064638473]]
    series_names = ['Sample-based RNN', 'Regular RNN']

    plot(x=x,
         ys=ys,
         series_names=series_names,
         title='MSE on Test Power Dataset',
         x_label='Sample Fraction',
         y_label='MSE',
         output_file_name=args.output_file_name)
