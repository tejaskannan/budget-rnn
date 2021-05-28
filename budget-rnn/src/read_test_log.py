import re
from argparse import ArgumentParser
from utils.file_utils import read_by_file_suffix

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True, help='Path to the test log.')
    args = parser.parse_args()

    test_log = list(read_by_file_suffix(args.test_log))[0]

    prediction_key_regex = re.compile('prediction[_-]*([0-9]*)')
    prediction_keys = []
    for key in test_log.keys():
        match = prediction_key_regex.match(key)
        if match is not None:
            level = int(match.group(1)) if len(match.groups()) > 1 else 0
            prediction_keys.append((key, level))

    for prediction_name, _ in sorted(prediction_keys, key=lambda t: t[1]):
        prediction_results = test_log[prediction_name]
        print('{0}: Accuracy -> {1:.5f}'.format(prediction_name, prediction_results['ACCURACY']))
