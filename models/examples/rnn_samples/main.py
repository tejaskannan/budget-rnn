
from utils.hyperparameters import HyperParameters
from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset


if __name__ == '__main__':
    hypers = HyperParameters('params.json')
    model = RNNSampleModel(hyper_parameters=hypers, save_folder='model')

    data_folder = 'small-input-dataset'
    dataset = RNNSampleDataset(data_folder, data_folder, data_folder)

    model.train(dataset=dataset)


