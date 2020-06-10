#!/bin/sh
#
#SBATCH --mail-user=tkannan@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tkannan/Documents/ml-models/src/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tkannan/Documents/ml-models/src/slurm/out/%j.%N.stderr
#SBATCH --partition=general
#SBATCH --job-name=adaptive-rnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12000

python train.py --data-folders datasets/har/folds_100_rounded --save-folder saved_models --params-files params/sample_params_fifth.json params/vanilla_params_fifth.json params/rnn_params.json
