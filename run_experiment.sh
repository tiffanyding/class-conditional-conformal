#!/bin/bash

# print some info for context
pwd | xargs -I{} echo "Current directory:" {}
hostname | xargs -I{} echo "Node:" {}

# Run all experiments in parallel. To run sequentially, remove "&"
for calibration_sampling in 'random';
    do for dataset in 'imagenet' 'cifar-100' 'places365' 'inaturalist'; 
        do for n in 10 20 30 40 50 75 100 150; 
            do python3 run_experiment.py $dataset $n -score_functions softmax APS RAPS -methods standard classwise cluster_random exact_coverage_standard exact_coverage_classwise exact_coverage_cluster --calibration_sampling $calibration_sampling -seeds 0 1 2 3 4 5 6 7 8 9 & 
        done; 
    done;
done


## Run a single experiment
# python run_experiment.py cifar-100 30 -score_functions softmax APS -methods standard always_cluster -seeds 0 1

