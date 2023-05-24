## Description of code organization
* `notebooks/` contains Jupyter notebooks for producing the figures and metrics from our paper. 
* `utils/` contains most the bulk of our code base. In particular, `utils/conformal_utils.py` contains implementations of standard, classwise, and clustered conformal (our proposed method). 


## To reproduce our experimental results
1. First, generate the softmax scores for each dataset. This requires obtaining and preprocessing the data, training a model, then applying the model to held out validation data. The code to do this is located in separate repositories. 
* For ImageNet, see my fork of `SimCLRv2-Pytorch` and follow the instructions in the first section of the README.
* For CIFAR-100, Places365, and iNaturalist, see `class-conditional-conformal-datasets` and follow the instructions in the README.

1. Update the file paths in the `load_dataset()` function in `experiment_utils.py` to point to the locations of the softmax scores and labels produced in the previous step. 

1. Run `sh run_experiment.sh` to generate results files.

1. Run the notebooks in `notebooks/` (update file paths if necessary).
