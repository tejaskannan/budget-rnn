from argparse import ArgumentParser

from utils.file_utils import read_by_file_suffix


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    print(read_by_file_suffix(args.file))
