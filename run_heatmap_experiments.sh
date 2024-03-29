#!/bin/bash

#SBATCH -N 1 # number of nodes requested
#SBATCH -n 20 # number of tasks (i.e. processes)
#SBATCH -t 0-12:00 # time requested (D-HH:MM)
#SBATCH -o /home/tding/slurm_output/heatmaps.%j.out # STDOUT
#SBATCH -e /home/tding/slurm_output _/heatmaps.%j.err # STDERR



# print some info for context
pwd | xargs -I{} echo "Current directory:" {}
hostname | xargs -I{} echo "Node:" {}

# Run all experiments
calibration_sampling='random'
dataset='imagenet'

for n in 10 50;
    do for frac_clustering in .1 .2 .3 .4 .5 .6 .7 .8 .9; 
        do for num_clusters in 2 3 4 5 6 8 10 15 20 50; 
            do save_folder=".cache/paper/heatmaps/${dataset}/frac=${frac_clustering}_numclusters=${num_clusters}"
            echo "Save folder: ${save_folder}" 
            python3 run_experiment.py $dataset $n -score_functions softmax APS RAPS -methods cluster_random --calibration_sampling $calibration_sampling -seeds 0 1 2 3 4 5 6 7 8 9 --frac_clustering $frac_clustering --num_clusters $num_clusters --save_folder $save_folder &
        done; 
    done;
done

