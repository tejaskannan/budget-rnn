from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from os.path import join
from typing import Optional

from utils.hyperparameters import extract_hyperparameters
from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset


def execute_predictions(model_name: str,
                        model_folder: RichPath,
                        data_folder: str,
                        output_folder: RichPath,
                        max_num_batches: Optional[int],
                        test_batch_size: int = 1):
    hypers_file = model_folder.join(f'model-hyper-params-{model_name}.pkl.gz')
    hypers = extract_hyperparameters(hypers_file)[0]

    model = RNNSampleModel(hyper_parameters=hypers, save_folder=model_folder)

    # Create dataset
    train_folder = join(data_folder, 'train')
    valid_folder = join(data_folder, 'valid')
    test_folder = join(data_folder, 'test')
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    model.restore_parameters(name=model_name)
    model.make(is_train=False)
    model.restore_weights(name=model_name)

    test_results = model.predict(dataset=dataset,
                                 name=model.name,
                                 test_batch_size=test_batch_size,
                                 max_num_batches=max_num_batches)

    print(test_results['latency'])

    test_result_file = output_folder.join(f'model-test-log-{model_name}.pkl.gz')
    test_result_file.save_as_compressed_file(test_results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--params-file', required=True, type=str)
    args = parser.parse_args()

    params_file = RichPath.create(args.params_file)
    assert params_file.exists(), f'The file {args.params_file} does not exist'

    params = params_file.read_by_file_suffix()
    model_folder = RichPath.create(params['model_folder'])
    output_folder = RichPath.create(params['output_folder'])

    output_folder.make_as_dir()

    num_models = len(params['models'])
    for i, model_name in enumerate(params['models']):
        print(f'Evaluating model {i+1}/{num_models}')
        execute_predictions(model_name=model_name,
                            model_folder=model_folder,
                            data_folder=params['data_folder'],
                            output_folder=output_folder,
                            max_num_batches=params['max_num_batches'],
                            test_batch_size=params['test_batch_size'])
