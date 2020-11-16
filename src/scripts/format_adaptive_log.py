from argparse import ArgumentParser

from utils.file_utils import read_by_file_suffix
from controllers.noise_generators import get_noise_generator


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True)
    parser.add_argument('--noise-scale', type=float, required=True)
    parser.add_argument('--noise-loc', type=float, required=True)
    args = parser.parse_args()

    noise_params = dict(loc=args.noise_loc, scale=args.noise_scale, noise_type='gaussian')
    noise_generator = list(get_noise_generator(noise_params=noise_params, max_time=0))[0]
    noise_key = str(noise_generator)

    test_log = list(read_by_file_suffix(args.log_file))[0]
    for budget, results in sorted(test_log[noise_key].items()):
        print('{0}: Accuracy -> {1:.3f}, Power: {2:.3f}'.format(budget, results['ACCURACY'], results['AVG_POWER']))
