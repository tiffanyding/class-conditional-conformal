#!/bin/bash

# Run this file using "sbatch my_script.sh"

# the SBATCH directives must appear before any executable
# line in this script

#SBATCH -n 64 # request CPUs
#SBATCH -t 0-48:00 # time requested (D-HH:MM)
# slurm will cd to this directory before running the script
# you can also just run sbatch submit.sh from the directory
# you want to be in
#SBATCH -D /home/tding/code/class-conditional-conformal-datasets/notebooks
# use these two lines to control the output file. Default is
# slurm-<jobid>.out. By default stdout and stderr go to the same
# place, but if you use both commands below they'll be split up
# filename patterns here: https://slurm.schedmd.com/sbatch.html
# %N is the hostname (if used, will create output(s) per node)
# %j is jobid
#SBATCH -o /home/tding/slurm_output/broader_scope_experiments_job=%j.out # STDOUT
#SBATCH -e /home/tding/slurm_output/broader_scope_experiments_job=%j.err # STDERR
# if you want to get emails as your jobs run/fail
##SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=tiffany_ding@eecs.berkeley.edu # Where to send mail
#seff $SLURM_JOBID
# print some info for context
pwd | xargs -I{} echo "Current directory:" {}
hostname | xargs -I{} echo "Node:" {}

# Run all experiments
calibration_sampling='random'
dataset='imagenet'

for n in 10 20 50 75;
    do for frac_clustering in .1 .2 .3 .4 .5 .6 .7 .8 .9; 
        do for num_clusters in 2 3 4 5 6 8 10 15 20 50; 
            do save_folder=".cache/paper/heatmaps/${dataset}/frac=${frac_clustering}_numclusters=${num_clusters}"
            echo "Save folder: ${save_folder}" 
            python run_experiment.py $dataset $n -score_functions softmax APS RAPS -methods cluster_random --calibration_sampling $calibration_sampling -seeds 0 1 2 3 4 5 6 7 8 9 --frac_clustering $frac_clustering --num_clusters $num_clusters --save_folder $save_folder &
        done; 
    done;
done

# for i in `find ./configs -name '*.yaml'` ; do python base_test.py $i & done


# Run a single experiment
# python run_experiment.py cifar-100 30 -score_functions softmax APS -methods standard always_cluster -seeds 0 1

