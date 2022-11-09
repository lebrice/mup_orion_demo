#!/bin/bash

sweep_job_id=$(sbatch --parsable sweep.sh)
training_script=sbatch --depend=afterany:$sweep_job_id --job-name=mup_large \
    train.sh
