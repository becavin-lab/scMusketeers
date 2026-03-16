#!/bin/bash

# Source the right python env
source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the python script
python3 "${SCRIPT_DIR}/benchmark_sbatch_review.py"
