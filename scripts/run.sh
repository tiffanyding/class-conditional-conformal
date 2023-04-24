#!/bin/bash

# Run this file using "sbatch my_script.sh"

# the SBATCH directives must appear before any executable
# line in this script

#SBATCH -t 0-48:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
#SBATCH -D /home/tding/code/class-conditional-conformal
# use these two lines to control the output file. Default is
# slurm-<jobid>.out. By default stdout and stderr go to the same
# place, but if you use both commands below they'll be split up
# filename patterns here: https://slurm.schedmd.com/sbatch.html
# %N is the hostname (if used, will create output(s) per node)
# %j is jobid
#SBATCH -o /home/tding/slurm_output/experiments_inaturalist_job=%j.out # STDOUT
#SBATCH -e /home/tding/slurm_output/experiments_inaturalist_job=%j.err # STDERR
# if you want to get emails as your jobs run/fail
##SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=tiffany_ding@eecs.berkeley.edu # Where to send mail
#seff $SLURM_JOBID
# print some info for context
python experiments_inaturalist.py
