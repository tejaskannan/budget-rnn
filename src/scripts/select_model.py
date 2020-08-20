import re
import os.path
import numpy as np
from argparse import ArgumentParser
from shutil import copyfile

from utils.file_utils import read_by_file_suffix, iterate_files, make_dir


TIMESTAMP_REGEX = re.compile('.*model-final-valid-log-[^0-9]+-([0-9\-]+)_model_best\.jsonl\.gz')


def select_model(input_folder: str, output_folder: str, model_type: str):
    """
    Function to find the model with the best validation accuracy and save all corresponding file to the output folder.
    """
    # Variables to track the model with the best accuracy
    best_timestamp = None
    best_accuracy = 0.0

    # We first iterate through the validation logs to select the model with the best
    # validation accuracy. This is how we choose models.
    for path in iterate_files(input_folder, pattern=r'model-final-valid-log-{0}.*\.jsonl\.gz'.format(model_type)):
        valid_log = list(read_by_file_suffix(path))[0]

        # Get the best validation accuracy
        accuracy = np.average([valid_log[prediction_name]['ACCURACY'] for prediction_name in valid_log.keys()])

        if accuracy > best_accuracy:
            match = TIMESTAMP_REGEX.match(path)
            best_timestamp = match.group(1)
            best_accuracy = accuracy

    # Create the output folder (if necessary)
    make_dir(output_folder)

    # We copy all files corresponding to the best timestamp into the output folder
    for path in iterate_files(input_folder, pattern=r'model-.*{0}.*'.format(best_timestamp)):
        file_name = path.split('/')[-1]
        output_file = os.path.join(output_folder, file_name)
        copyfile(path, output_file)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--model-type', type=str, required=True)
    args = parser.parse_args()

    select_model(input_folder=args.input_folder,
                 output_folder=args.output_folder,
                 model_type=args.model_type.upper())
