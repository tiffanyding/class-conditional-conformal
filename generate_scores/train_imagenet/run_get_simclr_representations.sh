#!/bin/bash

# Run this file using "sbatch my_script.sh"

# the SBATCH directives must appear before any executable
# line in this script

#SBATCH -p rise # partition (queue)
#SBATCH --cpus-per-task=72 # number of cores per task
# I think gpu:8 will request 8 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH -w como # Request como specifically
#SBATCH --exclude=freddie,flaminio,blaze # nodes not yet on SLURM-only
#SBATCH -t 0-48:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
#SBATCH -D /home/eecs/tiffany_ding/code/SimCLRv2-Pytorch
# use these two lines to control the output file. Default is
# slurm-<jobid>.out. By default stdout and stderr go to the same
# place, but if you use both commands below they'll be split up
# filename patterns here: https://slurm.schedmd.com/sbatch.html
# %N is the hostname (if used, will create output(s) per node)
# %j is jobid
#SBATCH -o /home/eecs/tiffany_ding/slurm_output/simclr_repr_train.out # STDOUT
#SBATCH -e /home/eecs/tiffany_ding/slurm_output/simclr_repr_train.err # STDERR
# if you want to get emails as your jobs run/fail
##SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=tiffany_ding@eecs.berkeley.edu # Where to send mail
#seff $SLURM_JOBID
# print some info for context
pwd | xargs -I{} echo "Current directory:" {}
hostname | xargs -I{} echo "Node:" {}
python get_simclr_representations.py train --batch_size=400
