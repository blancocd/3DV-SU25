#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1080ti:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=64gb
#SBATCH --output=train_correspondence.txt
#SBATCH --error=train_correspondence.txt

eval "$(~/miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
	export CONDA_INTERPRETER="3DVision-ex3" # Check if CONDA_INTERPRETER is not set, then set it to garage_2
fi
source activate "$CONDA_INTERPRETER"
which python3

python train_cor.py