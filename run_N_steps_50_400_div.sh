#!/bin/bash
#SBATCH --job-name dbelivan_ADCME_init_200
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --error=/home/dbelivan/research/run_N_steps_50_400_div.err
#SBATCH --output=/home/dbelivan/research/run_N_steps_50_400_div.out
#
#
echo "start script"
module --ignore-cache load "cuda/cuda-10.1"
nvcc --version
#
#
cd Optimization
/home/dbelivan/julia-1.5.3/bin/julia core_2.jl 400 50
#
echo "check end of script"
# end script