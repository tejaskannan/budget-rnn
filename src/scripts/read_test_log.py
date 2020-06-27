from argparse import ArgumentParser
from utils.file_utils import read_by_file_suffix

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    test_log = list(read_by_file_suffix(args.file))[0]
    for prediction_name, prediction_results in sorted(test_log.items()):
        print('{0}: Accuracy -> {1:.5f}'.format(prediction_name, prediction_results['ACCURACY']))
