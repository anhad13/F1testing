#!/bin/bash
#
#SBATCH -t72:00:00
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
python test_wsj.py  --config_file config/test.conf


