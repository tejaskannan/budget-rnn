import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Optional

from utils.file_utils import read_by_file_suffix
from plotting_constants import STYLE, LINEWIDTH

NAME_FORMAT = '{0} {1} {2}'


def plot_train_logs(train_log_files: List[str], labels: List[str], metric: str, output_file: Optional[str], max_num_epochs: Optional[int]):
    # Fetch training logs
    train_logs = list(map(read_by_file_suffix, train_log_files))

    with plt.style.context(STYLE):

        fig, ax = plt.subplots()

        for label, train_log in zip(labels, train_logs):
            for series, values in train_log[metric].items():
                if max_num_epochs is not None:
                    values = values[:max_num_epochs]

                epochs = list(range(len(values)))
                series_label = NAME_FORMAT.format(label.capitalize(), series.capitalize(), metric.capitalize())
                ax.plot(epochs, values, linewidth=LINEWIDTH, label=series_label)

                num_epochs = len(values)

        ax.set_xticks(list(range(0, num_epochs, 5)))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(series.capitalize())
        ax.set_title('Average {0} per Epoch'.format(metric.capitalize()))
        ax.legend()
 
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-logs', type=str, nargs='+')
    parser.add_argument('--labels', type=str, nargs='+')
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--max-num-epochs', type=int)
    parser.add_argument('--series', type=str, choices=['loss', 'accuracy'])
    args = parser.parse_args()

    # Validate arguments
    assert len(args.train_logs) == len(args.labels), f'The number of train logs must be equal to the number of labels'
    assert args.max_num_epochs is None or args.max_num_epochs > 0, 'The number of epochs must be positive'

    for log_file in args.train_logs:
        assert os.path.exists(log_file), f'The file {log_file} does not exist.'


    plot_train_logs(args.train_logs, args.labels, args.series, args.output_file, args.max_num_epochs)
