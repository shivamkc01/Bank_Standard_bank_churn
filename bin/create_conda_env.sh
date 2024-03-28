#!/bin/bash --login

# entire script fails if a single command fails


set -e

cd ..
PROJECT_DIR=$PWD

ENV_PREFIX="$PROJECT_DIR/env"
# Source Conda initialization script
source ~/miniconda3/etc/profile.d/conda.sh

conda activate base

conda env create --prefix "$ENV_PREFIX" --file "$PROJECT_DIR/environment.yml" --force

