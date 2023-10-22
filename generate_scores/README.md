# Datasets


## ImageNet

First, download ImageNet and update the `path` variable in `run()` (located in `train_imagenet/get_simclr_representations.py`). Then `cd` into the `generate_scores/train_imagenet` folder and run:

```
python download.py r152_3x_sk1
python convert.py r152_3x_sk1/model.ckpt-250228
python get_simclr_representations.py train 
python train_linear_and_get_softmax.py
```
(Code based on https://github.com/Separius/SimCLRv2-Pytorch)

## CIFAR-100

Download CIFAR-100 if necessary by running `sh download_cifar.sh`. 
Go to the `train_models` folder and open `cifar-100.ipynb`. Running all cells will train the model (if we have not already trained) and save the softmax scores and labels for the validation dataset will be saved to `train_models/.cache/`.

## Places365

Download the data from http://places.csail.mit.edu/index.html if necessary. Update the `root` argument of the `datasets.Places365()` dataloader in `get_dataloaders` (located in `train_models/torchvision_dataset_utils.py`) to point to the data.
Go to the `train_models` folder. If you work on a cluster with a SLURM scheduler, running `sbatch train_places365.sh` will train the model on the Places365 dataset. 0.1 of the dataset is reserved for validation. The softmax scores and labels for the validation dataset will be saved to `train_models/.cache/`.

## iNaturalist
Download the data from https://github.com/visipedia/inat_comp/tree/master/2021 if necessary. Update the `root` argument of the `datasets.INaturalist()` dataloader in `get_dataloaders` (located in `train_models/torchvision_dataset_utils.py`).
Go to the `train_models` folder. If you work on a cluster with a SLURM scheduler, running `sbatch train_inaturalist.sh` will train the model on the Places365 dataset. 0.5 of the dataset is reserved for validation. The softmax scores and labels for the validation dataset will be saved to `train_models/.cache/`.