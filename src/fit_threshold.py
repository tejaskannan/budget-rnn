from argparse import ArgumentParser
from utils.loading_utils import restore_neural_network
from controllers.power_utils import make_power_system, PowerType
from controllers.model_controllers import AdaptiveController
from dataset.dataset import DataSeries



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-paths', type=str, nargs='+', required=True, help='Paths to Budget RNN models.')
    parser.add_argument('--dataset-folder', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--budgets', type=float, nargs='+', required=True, help='Budgets to optimize for.')
    parser.add_argument('--precision', type=int, required=True, help='Optimization precision.')
    parser.add_argument('--population-size', type=int, default=1, help='The population size for population-based training.')
    parser.add_argument('--patience', type=int, default=25, help='The number of iterations without improving loss to tolerate before stopping.')
    parser.add_argument('--max-iter', type=int, default=100, help='The maximum number of optimization iterations.')
    parser.add_argument('--power-system-type', type=str, choices=['bluetooth', 'temp'], default='temp', help='The power spectrum to use.')
    parser.add_argument('--should-print', action='store_true', help='Whether to log information to sdtout during training.')
    args = parser.parse_args()

    for model_path in args.model_paths:
        print('Starting model at {0}'.format(model_path))

        # Create the power system
        model, _ = restore_neural_network(model_path, dataset_folder=args.dataset_folder)

        power_system = make_power_system(num_levels=model.num_outputs,
                                         seq_length=model.seq_length,
                                         model_type=model.model_type,
                                         power_type=PowerType[args.power_system_type.upper()])

        # Create the adaptive model
        controller = AdaptiveController(model_path=model_path,
                                        dataset_folder=args.dataset_folder,
                                        precision=args.precision,
                                        budgets=args.budgets,
                                        trials=args.population_size,
                                        patience=args.patience,
                                        max_iter=args.max_iter,
                                        power_system=power_system)

        # Fit the model on the validation set
        controller.fit(series=DataSeries.VALID, should_print=args.should_print)
        controller.save()
