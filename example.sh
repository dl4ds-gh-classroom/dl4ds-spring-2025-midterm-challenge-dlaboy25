#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -pe omp 4 # Request 4 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=6.0 # gpu compute capability
#$ -l h_rt=48:00:00 # Time limit
#$ -N job_name
#$ -j y # Merge standard output and error
#$ -o /output_folder

module load miniconda
conda activate dl4ds

python part1.py

### To submit this script, use the following command: qsub file_name.sh 