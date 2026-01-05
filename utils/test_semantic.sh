#!/bin/bash
#SBATCH --job-name=test-semantic
#SBATCH --output=./Log/test_semantic_output_%j.log
#SBATCH --error=./Log/test_semantic_error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --partition=x86_64_GPU
#SBATCH --gres=gpu:A800:1
#SBATCH --nodelist=gpu-n2

echo "===== Job started at: $(date) ====="
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Loading Anaconda..."

# Activate Conda environment
source /share/home/ecnuzwx/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

# Check Python path and version
echo "Using Python from: $(which python)"
python --version

# Check if script file exists
SCRIPT=test_semantic.py
if [ ! -f "$SCRIPT" ]; then
    echo "❌ Script $SCRIPT not found in current directory: $(pwd)"
    exit 1
else
    echo "✅ Found $SCRIPT"
    ls -lh $SCRIPT
fi

# Execute Python script
echo "===== Starting script: $SCRIPT ====="
python $SCRIPT
STATUS=$?

echo "===== Script exited with status: $STATUS ====="
if [ $STATUS -ne 0 ]; then
    echo "❌ Python script failed with status $STATUS"
else
    echo "✅ Python script ran successfully"
fi

# Record end time
echo "===== Job finished at: $(date) ====="