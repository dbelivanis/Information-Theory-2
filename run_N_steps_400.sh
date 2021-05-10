#!/bin/bash
#SBATCH --job-name dbelivan_ADCME_init_2000
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --error=/data/cees/dbelivan/check_core_1000_N_steps_400.err
#SBATCH --output=/data/cees/dbelivan/check_core_1000_N_steps_400.out
#
#
echo "start script"

module --ignore-cache load "cuda/cuda-10.1"
nvcc --version
#
#
cd Optimization

#module load julia/1.4.2

/home/dbelivan/julia-1.5.3/bin/julia core.jl 1000 400
#
echo "check end of script"
# end script
