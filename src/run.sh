#!/bin/sh
#
#SBATCH --mail-user=tkannan@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tkannan/Documents/ml-models/src/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tkannan/Documents/ml-models/src/slurm/out/%j.%N.stderr
#SBATCH --partition=fast
#SBATCH --job-name=adaptive-rnn-emg-phased
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12000

python train.py --data-folders datasets/emg/folds_50 --save-folder saved_models/emg --params-files params/emg/phased_rnn
