import re
from argparse import ArgumentParser
from typing import List

from utils.file_utils import read_by_file_suffix
from utils.testing_utils import ClassificationMetric


MODEL_TYPE_REGEX = re.compile(r'.*model-.*test-log-([^-]+)-.*')
HEADERS = ['Model', ClassificationMetric.ACCURACY.name, ClassificationMetric.PRECISION.name, ClassificationMetric.RECALL.name, ClassificationMetric.F1_SCORE.name, ClassificationMetric.FLOPS.name]


def get_model_type(test_log_file: str) -> str:
    match = MODEL_TYPE_REGEX.match(test_log_file)
    return match.group(1)


def create_table(test_log_files: List[str], digits: int, output_file: str):
    
    # Extract data
    result: Dict[str, Dict[str, float]] = dict()
    for test_log_file in test_log_files:
        test_log = list(read_by_file_suffix(test_log_file))[0]
        model_type = get_model_type(test_log_file)

        for series, model_results in test_log.items():
            if not isinstance(model_results, dict):
                continue

            series_name = series.replace('_', ' ').replace('prediction', 'level')
            series_name = ' '.join([t.capitalize() for t in series_name.split(' ')])
            series_name = '{0} {1}'.format(model_type.capitalize(), series_name)

            result_dict: Dict[str, float] = dict()
            for metric in ClassificationMetric:
                metric_value = model_results[metric.name]
                if metric == ClassificationMetric.LATENCY:
                    metric_value *= 1000.0

                result_dict[metric.name] = metric_value

            result[series_name] = result_dict

    # Create table rows
    rows: List[str] = []
    for series, series_dict in sorted(result.items()):
        row: List[str] = [series]
        for header in HEADERS[1:]:
            value = str(round(series_dict[header], digits))
            row.append(value)

        rows.append(' & '.join(row))

    # Convert to Latex table
    headers: List[str] = []
    for h in HEADERS:
        h = h.replace('_', ' ')
        h = ' '.join([t.capitalize() for t in h.split(' ')])
        headers.append(f'\\textbf{{{h}}}')
    headers_str = ' & '.join(headers) + ' \\\\\n\\midrule'

    table_rows: List[str] = ['\\begin{tabular}{lccccc}', headers_str]
    table_rows.append(' \\\\\n'.join(rows))
    table_rows.append('\\end{tabular}')

    table_str = '\n'.join(table_rows)

    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(table_str)
    else:
        print(table_str)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-logs', type=str, nargs='+')
    parser.add_argument('--digits', type=int, default=3)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    create_table(args.test_logs, args.digits, args.output_file)
