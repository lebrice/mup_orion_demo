#!/bin/bash

sweep_job_id=$(sbatch --parsable run_sweep.sh)
sbatch --depend=afterany:$sweep_job_id --gres=gpu:rtx8000:2 --job-name=mup_large_2gpu train_big_model.sh
sbatch --depend=afterany:$sweep_job_id --gres=gpu:rtx8000:4 --job-name=mup_large_4gpu train_big_model.sh
