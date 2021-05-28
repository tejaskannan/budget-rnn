# Budget RNNs: Multi-Capacity Neural Networks to Improve In-Sensor Inference under Energy Budgets

This repository holds the relevant code for Budget RNNs, a system for performing RNN inference under energy budgets on low-power sensors. This work was published in RTAS 2021.

There are three components of this repository: neural network training, halting threshold optimization, and simulating budget inference. This document describes how to execute each of these components. The provided code reproduces the simulation results from the Budget RNN paper (Sections 5.B through 5.F). The hardware experiments (Section 5.G) require a TI MSP 430 FR5994 MCU, an HM-10 BLE device, and a set of supercapacitors. We are not able to provide access to this setup over the Internet. If you have these components and want to reproduce the hardware experiments, we can provide the relevant code.

As a general note, the code often makes references to a model named Sample RNN. This name was the old name for Budget RNNs. Thus, within this codebase, Sample RNNs and Budget RNNs are equivalent.

## Installation
The repository has a few dependencies on popular python libraries. To install the relevant packages, navigate into the `budget-rnn` directory and execute the shell script `install.sh` from the command line. That is, from the directory `budget-rnn`, run the following.
```
$ ./install.sh
```
This script creates a python 3.6 virtual environment called `budget-rnn-env` and installs all packages (below) within this virtual environment.
| Package                |
| ---------------------- |
| tensorflow (v2.2)      |
| numpy                  |
| matplotlib             |
| more_itertools         |
| scipy                  |
| scikit-learn (v0.22)   |

The only pre-requisites for this script are `python3.6` (or higher) and `pip3` (found [here](https://pip.pypa.io/en/stable/)).

After the script finishes, enter the virtual environment using the command below. You must execute this command from the base `budget-rnn` directory.
```
$ source budget-rnn-env/bin/activate
```
You must run all python scripts within this virtual environment. **If you close your terminal, you must re-enter the virtual environment using the above command.** To leave the virtual environment, execute the following.
```
$ deactivate
```
As a note, the installation script will make a system-wide install of the python package `virtualenv`. If you want to remove this python package after executing Budget RNNs, you may run the following.
```
$ pip3 uninstall virtualenv
```
Finally, this installation process was tested on Linux. If running on Windows, please use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

## Directory Structure
The repository uses the structure below. Each submodule contains its own documentation. Most code files are located in the `budget-rnn/src` directory.
1. `data/`: This folder contains the datasets used for training and evaluation. Each dataset has a folder entitled `<dataset-name>/folds_<N>` where `<dataset-name>` is the name of the dataset and `<N>` is the sequence length. The `folds_<N>` folder has three sub-directories containing the `train`, `validation`, and `test` folds.
2. `results/`: This folder contains the results of budgeted inference obtained in the simulated environment. The repository includes results from both the Bluetooth and the Temperature energy profiles.
3. `trained_models/`: This folder holds already-trained models on each dataset. The Budget RNNs come with trained halting thresholds.
4. `budget-rnn/src/c_implementation`: This folder contains a C implementation of the considered RNNs. This implementation is used for debugging the MSP430 implementation on a standard system.
5. `budget-rnn/src/controllers/`: This directory holds implementations of the runtime systems used to select subsequences during budgeted inference.
6. `budget-rnn/src/conversion/`: This folder holds utility files for converting Tensorflow models into C header files for MSP430 inference.
7. `budget-rnn/src/dataset/`: This directory contains code that manages data loading.
8. `budget-rnn/src/layers/`: This folder holds the implementation of neural network layers. The 'layers/cells' folder contains each RNN cell.
9. `budget-rnn/src/models/`: This folder contains the implementation of each neural network. In particular, the `adaptive_model.py` file describes the logic for Budget RNNs. The `standard_model.py` file implements the baseline RNNs. The `tf_model.py` file contains the logic to train and execute models within Tensorflow.
10. `budget-rnn/src/msp`: This module contains the MSP430 implementation. This code will not run on a standard system (e.g. laptop).
11. `budget-rnn/src/params/`: This directory holds the parameter files that govern each neural network. The provided parameter files were used to train the existing RNNs.
12. `budget-rnn/src/results/`: This folder contains the results of budgeted inference obtained in the simulated environment. The repository includes results from both the Bluetooth and the Temperature energy profiles.
13. `budget-rnn/src/utils/`: This module provides useful utility functions such as file reading and writing. Of particular interest is the `utils/tf_utils.py` file which implements utility Tensorflow operations.

## Overview
This document starts by describing how to recreate the plots in the paper using the raw data from already-run simulations. The following sections describe how to train Budget RNNs and execute new simulations from scratch. This end-to-end process may be time-consuming. For this reason, we include already-trained neural networks, optimized halting thresholds, and traces from the simulations used in the paper. Thus, one may (partially) validate the final results without completing each step. All scripts can be found in the folder `budget-rnn/src`, and you should run the scripts from this directory. The existing results and models are located in the base of the repository.

## Generating Results
This section covers how to generate the main results in the paper. The folders `results/<sensor-name>/merged/<dataset-name>` contain the raw simulation results using the included RNNs. If you are generating results for new simulations, alter the arguments in the following commands to point to the new simulation logs.

### Budget vs Accuracy Curves
The first component involves plotting the accuracy for each considered budget. The script `plot_budget_curve.py` implements this routine. This script takes the following arguments.
1. `--input-folder`:  Path to the folder containing the (merged) simulation results. This path should be the output folder of the `merge_systems.py` script.
2. `--output-file`: An *optional* output file to save the plot. If none, then the script will show the plot.
3. `--noise-loc`: The noise bias for which to get the accuracy values.
4. `--dataset-folder`: Path to the dataset corresponding to the simulation results.

As an example, the following command will recreate figure 6 in the paper.
```
$ python plot_budget_curve.py --input-folder ../../results/bluetooth/merged/pen_digits/ --noise-loc 0.0 --dataset-folder ../../data/pen_digits/folds_8/
```
### Accuracy Tables
The second component is to compute the aggregate accuracy across all budgets for each system. The script `accuracy_table.py` implements this behavior. This script takes the following arguments.
1. `--input-folders`: Paths to folders containing the merged simulation results. These results may be from multiple datasets.
2. `--mode`: The comparison mode in `{baseline, budget}`. The mode specifies whether to compare to baseline RNNs or Budget RNN variations.
3. `--noise-loc`: The noise bias.

For example, the following command will recreate the left-hand side of figure 5.
```
$ python accuracy_table.py --input-folders ../../results/bluetooth/merged/emg/ ../../results/bluetooth/merged/fordA/ ../../results/bluetooth/merged/pavement/ ../../results/bluetooth/merged/pedestrian/ ../../results/bluetooth/merged/pen_digits/ ../../results/bluetooth/merged/uci_har/ ../../results/bluetooth/merged/whale --mode baseline --noise-loc 0.0
```
Switching the Bluetooth results for the temperature results will recreate the right half of this figure. Furthermore, changing the `--mode` to `budget` will recreate figure 9.

The accuracy table can also recreate figure 8 by applying nonzero noise biases. The 10% and 20% noise biases corresponding to the following noise `loc` values on each sensor.

| Sensor       | +20% | +10% | -10% | -20% |
| ------------ | :--: | :--: | :--: | :--: |
| Bluetooth    | 2.0  | 1.0  | -1.0 | -2.0 |
| Temperature  | 0.4  | 0.2  | -0.2 | -0.4 |
Figure 8 results from taking the `All` row in the accuracy table of the above biases.

### Energy Comparison Plots
The final component of result generation involves the mean energy budget comparison. The script `plot_energy.py` creates a bar chart showing the mean normalized energy budget required to accuracy equivalent to the Budget RNN. This script has the following argument.
1. `--input-folders`: Paths to folders containing the merged simulation results. These results may be from multiple datasets.

As an example, the following command will recreate figure 7 from the paper.
```
python plot_energy.py --input-folders ../../results/bluetooth/merged/emg/ ../../results/bluetooth/merged/fordA/ ../../results/bluetooth/merged/pavement/ ../../results/bluetooth/merged/pedestrian/ ../../results/bluetooth/merged/pen_digits/ ../../results/bluetooth/merged/uci_har/ ../../results/bluetooth/merged/whale
```

### Training Time
The script `training_time.py` uses the neural network training logs to print out the training time. For Budget RNNs, this time accounts for the time to fit the halting thresholds. The script takes the following parameter.
1. `--input-folder`: Path to the folder containing the model training logs. All models in this folder should be the same type. The training time will be the sum of the times from each model.

For example, the command below will print the time to train all Budget RNNs on the Pen Digits dataset.
```
$ python training_time.py --input-folder trained_models/pen_digits/budget
```
Changing the dataset and model type will recreate the training time results in Figure 10.

The script `training_iters.py` measures the approximate training iterations for each RNN type. For Budget RNNs, the number of iterations considers the halting threshold optimization. The script takes the arguments below.
1. `--input-folder`: Path to the folder containing the model training logs. All models in this folder should be the same type. The training time will be the sum of the times from each model.
2. `--dataset-folder`: Path to the dataset folder used for training and evaluation.

For example, the command below will print the number of training iterations for the Budget RNNs on the Pen Digits dataset.
```
$ python training_iters.py --input-folder trained_models/pen_digits/budget/ --dataset-folder data/pen_digits/folds_8/
```
By altering the dataset and model type, you can recreate the training iteration results in Figure 11.

As a note, both of these scripts use logged information from the original training runs. The neural network training metrics are logged in the `TFModel.train()` function of `models/tf_model.py`. The function `AdaptiveController.fit()` in `controllers/model_controller.py` logs the cost of halting threshold optimization. The script `training_iters.py` measures the approximate training iterations for each RNN type. For Budget RNNs, the number of iterations considers the halting threshold optimization. The script takes the arguments below.
1. `--input-folder`: Path to the folder containing the model training logs. All models in this folder should be the same type. The training time will be the sum of the times from each model.
2. `--dataset-folder`: Path to the dataset folder used for training and evaluation.

For example, the command below will print the number of training iterations for the Budget RNNs on the Pen Digits dataset.
```
$ python training_iters.py --input-folder trained_models/pen_digits/budget/ --dataset-folder data/pen_digits/folds_8/
```
By altering the dataset and model type, you can recreate the training iteration results in Figure 11.

As a note, both of these scripts use logged information from the original training runs. The neural network training metrics are logged in the `TFModel.train()` function of `models/tf_model.py`. The function `AdaptiveController.fit()` in `controllers/model_controller.py` logs the cost of halting threshold optimization.

## Neural Network Training
The codebase trains all neural networks in Tensorflow. In this section, we describe how to train Budget RNNs using the provided code.

### Training
The script `src/train.py` drives the training process. This script takes the following five parameters.
1. `--data-folders`: Paths to dataset folders e.g. `data/pen_digits/folds_8`.
2. `--params-files`: Paths to the model parameter files e.g. `params/pen_digits/budget/budget_4.json`.
3. `--save-folder`: Directory in which to save the trained model.
3. `--trials`: An *optional* parameter that specifies the number of times to train the given model. The default is `1`.
4. `--should-print`: A boolean flag denoting whether to print logging information during training.
5. `--testrun`: A boolean flag that caps the execution to a single epoch. This flag is useful for debugging.

As an example, we can train a Budget RNN on the `pen_digits` dataset (for a single epoch) using the following command (run from the `budget-rnn/src` directory).
```
$ python train.py --data-folders data/pen_digits/folds_8 --params-files params/pen_digits/budget/budget_4.json --save-folder saved_models --should-print --testrun
```
The code will place the saved model in the folder `saved_models/<dd_mm_yyyy>` where `<dd_mm_yyyy>` is the current date. After completing training, there will be six output files as described below.
1. `model-<model-name>-<dataset-name>-<date>_model_best.pkl.gz`: The trained model parameters.
2. `model-metadata-<model-name>-<dataset-name>-<date>_model_best.pkl.gz`: The model's metadata. Among other information, this file contains the data normalizers.
3. `model-hyper-params-<model-name>-<dataset-name>-<date>_model_best.pkl.gz`: The model hyperparameters.
4. `model-train-log-<model-name>-<dataset-name>-<date>_model_best.pkl.gz`: A log of the training and validation accuracy for each epoch.
5. `model-final-valid-log-<model-name>-<dataset-name>-<date>.jsonl.gz`: The model accuracy on the validation set.
6. `model-test-log-<model-name>-<dataset-name>-<date>_model_best.jsonl.gz`: The model accuracy on the test set.

In each of these file names `<model-name>` is the name of the model (e.g. `BUDGET_RNN`), `<dataset-name>` is the name of the dataset (e.g. `pen-digits`), and `<date>` is the time of training (down to the second). As a note, if you halt training early (e.g. using `Ctrl-C`), the code will not create training, validation, and testing logs.

The validation and testing logs contain metrics such as accuracy and macro F1 score. Many of the considered models have more than one output. In these cases, the logs store the metrics for each output under the key `prediction_L` where `L` is the output level (`0`-indexed). The script `read_test_log.py` prints out the per-level accuracy from these logs. To run this script, use the command below (from the `budget-rnn/src` directory) after filling in the model name, dataset, and date.
```
$ python read_test_log.py --test-log saved_models/<dd>_<mm>_<yyyy>/model-test-log-<model-name>-<dataset-name>-<date>_model_best.jsonl.gz
```
The terms `<dd>`, `<mm>`, and `<yyyy>` should be the day, month, and year of model training. This command assumes the `--save-folder` during training is `saved_models`. If using a provided model, use `trained_models` instead.

In general, it may take hours to fully train a single model. Furthermore, each training run may yield slightly different results due to random initializations.

### Testing
The script `test.py` drives model testing. As the `train.py` script will already execute testing after model training, this script is mainly used to re-test models. The script takes the following arguments.
1. `--model-path`:  Path to the trained model parameters file.
2. `--dataset-folder`: Path to the dataset to execute testing on.
3. `--batch-size`:  An *optional* test batch size. The default value is the training batch size.
4. `--max-num-batches`: An *optional* maximum number of batches to execute. Useful for debugging.
5. `--series`: The series in `{train, valid, test}` to execute on. The default is `test`.

As an example, we can execute the following concrete command from the `budget-rnn/src` directory.
```
$ python test.py --model-path trained_models/pen_digits/budget/model-SAMPLE_RNN-pen-digits-2020-09-16-17-50-15_model_best.pkl.gz --dataset-folder data/pen_digits/folds_8/
```
The results will be saved in a log file in the same directory as the provided model. This log file has the same structure as the previously described logs. As a note, if a log is already present, then this command will overwrite the previous information.

## Halting Threshold Optimization
To perform budgeted inference, Budget RNNs use thresholds on their halting signals. The script `fit_thresholds.py` trains these thresholds for a discrete set of budgets. This script takes the following arguments.
1. `--model-paths`: Paths to Budget RNN models on which to fit thresholds. **Models other than Budget RNNs will cause this script to fail.**
2. `--dataset-folder`: Path to the dataset. This dataset should be the same as the training dataset.
3. `--budgets`: A list of budgets on which to fit thresholds. To standardize across datasets, provide the budget in terms of power (mW). The energy budget is then `power_budget * sample_period * seq_length * num_sequences`.
4. `--precision`: The precision  (`k`) of threshold values. Each iteration will optimize over `2^k` threshold values. 
5. `--population-size`:  The population size for population-based training.
6. `--patience`: The number of iterations without improving loss to tolerate before early stopping.
7. `--max-iter`: The maximum number of optimization steps.
8. `--power-system-type`: The sensor type in `{bluetooth,temp}`.
9. `--should-print`: Whether to print information to sdtout during training.

As an example, we can fit thresholds for a Budget RNN on the `pen_digits` task using energy corresponding to a temperature sensor. The command below initiates this optimization.
```
$ python fit_threshold.py --model-paths trained_models/pen_digits/budget/model-SAMPLE_RNN-pen-digits-2020-09-16-17-50-15_model_best.pkl.gz --dataset-folder data/pen_digits/folds_8/ --budgets 1.0 --precision 8 --population-size 4 --patience 5 --max-iter 10 --power-system-type temp --should-print
```
The tuned thresholds will be stored in a file named `model-controller-<sensor>-BUDGET_RNN-<dataset>-<date>_model_best.pkl.gz` in the same folder as original model. **The code will overwrite any existing controllers**. The folders `trained_models/<dataset/budget/controllers` contain backups of the original controllers.

In the `trained_models` folder, the Budget RNNs all have optimized thresholds. The provided thresholds use the following hyperparameters. These settings hold for both sensing profiles. The optimization script fixed the random seed, so the results should be stable across independent trials.
| Parameter       | Value |
| --------------- | :---: |
| Precision       | 8     |
| Population Size | 10    |
| Patience        | 25    |
| Max Iters       | 50    |

For each sensor type, the existing Budget RNNs have thresholds for the following budgets. Note that the `pen_digits` dataset differs from the other datasets due to a shorter, non-multiple sequence length. Each range is inclusive on both ends.
| Dataset         | Bluetooth                       | Temperature                            |
| --------------- | :-----------------------------: | :------------------------------------: |
| Pen Digits      | Low: `6`, High: `16`, Step: `1` | Low: `1.0`, High: `2.75`, Step: `0.25` |
| All Others      | Low: `4`, High: `16`, Step: `1` | Low: `0.5`, High: `2.75`, Step: `0.25` |

Training all halting thresholds to completion may take a few hours.

## Budgeted Inference Simulation
The script `simulator.py` executes budgeted inference in a simulated environment. The simulator relies on a simulated power system, as well as model controllers. These components are found in the `controllers/` directory. The simulator script takes the following arguments.
1. `--adaptive-model-paths`: Paths to Budget RNNs model parameters.
2. `--baseline-model-path`: Path to the standard RNN model parameters.
3. `--skip-model-folder`: Path to the Skip RNN folder.
4. `--phased-model-folder`: Path to the Phased RNN folder.
5. `--dataset-folder`: Path to the dataset folder. All models should be trained on the same dataset.
6. `--budget-start`:  Smallest budget to execute (inclusive).
7. `--budget-end`: Largest budget to execute (inclusive).
8. `--budget-step`: Step between consecutive budgets.
9. `--output-folder`: Path to store output results.
10. `--noise-params`: Noise parameter files. These may be found in `controllers/noise_params`.
11. `--power-system-type`: Simulated sensor type in `{bluetooth, temp}`.
12. `--skip-plotting`: Whether to skip plotting the time series results.
13. `--save-plots`: Whether to save the time series plots.

The scripts `simulate_temp.sh` and `simulate_bt.sh` execute the simulator on all trained models. To run these scripts, use the following syntax.
```
$ ./simulate_temp.sh <output-folder>
```
These scripts provide examples of how to use the simulator. As a note, the simulator will automatically use the Budget RNN thresholds present in the same directory. Thus, you may want to restore the thresholds to the backup in the `trained_models/<dataset-name>/budget/controllers/` directory.

The noise parameter files control the noise levels. The folder `controllers/noise_params` contains two default parameters for each sensor type. In these parameters, the `loc` field specializes the noise bias. When given a list, the simulator will run independent experiments for each provided bias.

The simulator stores each result in the given output folder. Each (RNN, control policy) paring gets its own simulation log. The log is a dictionary mapping the noise type to an inner dictionary; the inner dictionary maps the budget to the system's results. The folders `results/bluetooth` and `results/temperature` contain the results from the provided models.  

### Merging Systems
Budget RNNs select models based on the current budget. The script `merge_systems.py` merges the simulation results using model selection derived from the Budget RNN validation accuracy. This script takes the following arguments.
1. `--models`:  Paths to the Budget RNN model parameters. This argument can be a directory containing the models.
2. `--dataset-folder`: The path to the dataset. This folder should be the same dataset as used during training.
3. `--log-folder`: Path to folder containing the simulation logs. This path denotes the budgeted inference simulation output for this dataset.
4. `--power-system-type`: The sensor type in `{bluetooth, temp}`. This sensor should be the same type as used in the simulator.
5. `--output-folder`: Path to the output folder.

The script merges the logs from the Budget RNN systems. Further, it will copy the other systems' simulator results into the output folder. Thus, the provided output folder will contain the results for all systems.

As an example, the command below will merge the logs on the pen digits dataset.
```
$ python merge_systems.py --models ../../trained_models/pen_digits/budget/ --dataset-folder ../../data/pen_digits/folds_8/ --log-folder ../../results/bluetooth/pen_digits/ --output-folder ../../results/pen_digits_merged --power-system-type bluetooth
```
The merged logs from the already-trained models are located in the `results/<sensor-name>/merged/<dataset-name>` directories.

### Energy Budget Comparison
To add context to the accuracy differences, we can compare the amount the baseline budgets must change to obtain accuracy equivalent to the Budget RNN. The script `energy_comparison.py` implements this functionality. For each baseline accuracy result, the code uses a binary-search technique to find the budget on which the Budget RNN achieves comparable accuracy. The difference in the baseline budget and the Budget RNN budget forms the comparison. The script uses the following parameters.
1. `--adaptive-model-paths`:  Paths to the Budget RNN model parameters. This parameter can be a directory containing the models.
2. `--adaptive-log`:  Path to the merged Budget RNN simulation results.
3. `--baseline-log`: Paths to the baseline simulation results.
4. `--dataset-folder`:  Path to the corresponding dataset folder.
5. `--sensor-type`: The sensor type in `{bluetooth,temp}`. This setting should align with the simulation results.
6. `--should-print`: Whether to print to stdout during execution.

The script will create a file called `energy_comparison.jsonl.gz` the same directory as the (merged) Budget RNN simulation log. This action will overwrite any existing files. The folders `results/<sensor>/merged/<dataset>/comparison` contain backups of the energy comparison logs.

The command below will compare the Budget RNN to the standard RNN on the pen digits dataset. This comparison uses the Bluetooth profile on the existing simulation results. As a note, this script can take a long time to execute.
```
python energy_comparison.py --adaptive-model-paths ../../trained_models/pen_digits/budget/ --adaptive-log ../../results/bluetooth/merged/pen_digits/model-adaptive-SAMPLE_RNN-pen_digits-merged-bluetooth.jsonl.gz --baseline ../../results/bluetooth/merged/pen_digits/model-fixed_under_budget-RNN-pen-digits-2020-08-28-23-24-40_model_best-bluetooth.jsonl.gz --dataset-folder ../../data/pen_digits/folds_8/ --sensor-type bluetooth --should-print
```
Adding additional baseline logs (e.g. Skip and Phased) will expand this comparison. Furthermore, changing `bluetooth` to `temp` will compare systems on the temperature sensor profile.

## MSP430 Experimentation

There are three main steps involved with getting the implemented RNNs to run on a TI MSP430 FR5994: (1) model conversion, (2) dataset conversion, (3) loading onto the device. The sections below cover each part.

### Neural Network Conversion
To run on the MSP430, the neural network parameters must be converted into a C header file. This process is accomplished by the script `convert_network.py` which takes the following arguments.
1. `--model-path`: Path to the trained RNN model. This must be the pickle file containing the trained parameters.
2. `--sensor-type`: The sensor type (`bluetooth` or `temperature`). This parameter is used to obtain the halting thresholds and the offline power profile.
3. `--precision`: The fixed point precision (in bits). For example, a precision of `9` means that `9` of the `16` bits per value will be fractional (`5` will not non-fractional and `1` will be the sign).
4. `--msp`: Whether to prepare the model for the MSP430. If not set, then the model can be run on a standard system using the provided `c_implementation`.

The output of this script is a file called `neural_network_parameters.h`. You should copy this file into the folder containing the model's C implementation.

### Dataset Conversion
We facilitate computation by pre-quantizing each dataset. The script `create_mcu_dataset.py` performs this quantization and takes in the following two parameters.
1. `--data-folder`: The folder of the dataset to prepare. The script will automatically use the testing fold.
2. `--precision`: The number of fractional bits used during fixed point quantization. This should match the quantization applied to the model.

The outputs will be written to two files placed in the dataset folder: `test_<N>_inputs.txt` and `test_<N>_labels.txt` where `<N>` is the `precision`.

### MSP430 Execution
The folder `msp` contains an implementation of each RNN designed for an TI MSP430 FR5994. To run this code, you must copy the `neural_network_parameters.h` file into this folder, compile the code, and load it onto the MSP device.

Once the MSP430 is ready, you can use the script `sensor_client.py` to send inputs to the MSP430 and have it respond with inference results. This experimental setup uses the Bluetooth link as a `sensor` so allow for testing with pre-collected data. This script takes the following parameters.
1. `--sample-freq`: The sample frequency in Hz
2. `--seq-length`: The sequence length of this dataset
3. `--max-sequences`: The maximum number of sequences to test. This parameter must match the corresponding constant in the MSP430 code.
4. `--inputs-path`: The path to the quantized input data (txt) file. This file is one output of the data conversion step above.
5. `--labels-path`:  The path to the label (txt) file. This file is the other output of the data conversion step above.
6. `--output-file`: Path to the output (jsonl.gz) file. The results will be stored in this file.
7. `--budget BUDGET`: The energy budget. This parameter must match the corresponding constant in the MSP430 code.
8. `--is-skip`: Whether this script is used to drive a Skip RNN.
9. `--start-index`: The optional element to start inference at. This parameter defaults to `0`.

This client will communicate with the MSP430 over a Bluetooth Low Energy link. The MAC address and BLE device are specified at the top of the `sensor_client.py`; these values should be changed to match the employed hardware.
