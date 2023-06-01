
<!-- TODOs:
[ ] rewrite model training
[ ] Run notebooks using clean environment: or open in colab  -->

This is the code release accompanying the paper [TODO: add arXiv link]

Citation: 
```
@article{ding2021class,
  title={Class-Conditional Conformal Prediction with Many Classes},
  author={Ding, Tiffany and Angelopoulos, Anastasios N and Bates, 
          Stephen and Jordan, Michael I and Tibshirani, Ryan J},
  journal={TODO},
  year={2023}
}
```


## Setup 

First, create a virtual environment and install the necessary packages by running

```
conda create --name env
conda activate env
pip install -r requirements.txt
```

To make the environment accessible from Jupyter notebooks, run

```
ipython3 kernel install --user --name=conformal_env
```

This adds a kernel called `conformal_env` to your list of Jupyter kernels.

Download the datasets by running

```
sh download_data.sh
```

which will create a folder called `data/` and download the data described in the following section. 

## Data description

1. `imagenet` (4.62 GB): `(115301, 1000)` array of softmax scores and `(115301,)` array of labels
1. `cifar-100` (0.01 GB): `(30000, 100)` array of softmax scores and `(30000,)` array of labels
1. `places365` (0.54 GB): `(183996, 365)` array of softmax scores and `(183996,)` array of labels
1. `inaturalist` (6.72 GB): `(1324900, 633)` array of softmax scores and `(1324900,)` array of labels

The code for training models on the raw datasets to produce the softmax scores is located in `generate_scores/`

## Running Clustered Conformal

See `example.ipynb` for an example of how to run clustered conformal prediction. 

## Reproducing our experiments

Run `sh run_experiments.sh` to run our main set of experiments. Run `sh run_heatmap_experiments.sh` for experiments that test the sensitivity of clustered conformal to the hyperparameter values. To view the main results, run `jupyter notebook` from Terminal, then run the notebooks in the `notebooks/` directory.

