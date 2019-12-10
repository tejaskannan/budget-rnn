import numpy as np
from argparse import ArgumentParser
from dpu_utils.utils import RichPath


def find_best_model(model_type: str, dataset: str, model_folder: RichPath) -> str:

    best_loss = 1e7
    best_model = None

    model_filter = f'model-train-log-{model_type}_rnn_model-{dataset}-*.pkl.gz'
    for model_train_log in model_folder.iterate_filtered_files_in_dir(model_filter):
        train_log = model_train_log.read_by_file_suffix()

        best_model_loss = 1e7
        for loss_dict in train_log:
            valid_losses = loss_dict['valid_losses']
            avg_valid_loss = np.average(list(valid_losses.values()))

            if avg_valid_loss < best_model_loss:
                best_model_loss = avg_valid_loss

        print(model_train_log)
        print(best_model_loss)
        if best_model_loss < best_loss:
            best_loss = best_model_loss

            model_train_log_name = model_train_log.path.split('/')[-1]
            model_name = model_train_log_name.replace('-train-log-', '-').replace('.pkl.gz', '')
            best_model = model_name

    return best_model



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-types', type=str, nargs='+')
    parser.add_argument('--datasets', type=str, nargs='+')
    parser.add_argument('--model-folder', type=str, required=True)
    args = parser.parse_args()

    model_folder = RichPath.create(args.model_folder)
    assert model_folder.exists(), f'The folder {model_folder} does not exist!'

    for model_type in args.model_types:
        for dataset in args.datasets:
            best = find_best_model(model_type, dataset, model_folder)
            print(f'Best model for {model_type} and {dataset}: {best}')
