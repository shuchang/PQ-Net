#!/bin/bash --login
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=v100:2
#SBATCH --cpus-per-gpu=4  
#SBATCH --mem=64G 
#SBATCH --partition=batch 
#SBATCH --mail-type=ALL
#SBATCH --output=results/%x/%j-slurm.out
#SBATCH --error=results/%x/%j-slurm.err

# activate the conda environment
conda activate base

# python data/sample_points_from_voxel.py --src data --category Chair

#################### Training ####################

# # train transformer model
python train.py --module transformer \
                --data_root data \
                --category Chair \
                --nr_epochs 1000 \
                --batch_size 64 \
                --lr 1e-3 \
                --save_frequency 500 \
                -g 0 \
                --vis \


#################### Testing ####################

python test.py --rec \
                --module transformer \
                --data_root data \
                --category Chair \
                --ckpt 1200 \
                --format voxel \
                --by_part True
