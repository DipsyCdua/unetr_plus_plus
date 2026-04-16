#!/bin/bash
set -e
echo "run prepare dataset"
python3 prepare_dataset.py "$@"
