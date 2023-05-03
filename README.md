# Associated Code for Paper "A Toy Model of Universality: Reverse Engineering how Networks Learn Group Operations"

## Setup

To run the code, create a virtual environment and install the requirements listed under `requirements.txt`. The code was tested with Python 3.8.10.

Optionally download and unzip a cached file of useful tensors by running `download_cache.sh`. This will download a large file from google drive and extract it to the correct location. This is not necessary to run the code, but will speed up the code by caching some large tensors.

## Demo Notebook

<a target="_blank" href="https://colab.research.google.com/github/bilal-chughtai/rep-theory-mech-interp/blob/master/rep_theory/paper/demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

We have included a standalone demo notebook, `rep_theory/paper/demo.ipynb`, which demonstrates the main ideas of the paper. This notebook is mostly self-contained, and does not extensively use any other code in the repository. Read this notebook first to get a sense of the main ideas in the paper, and their implementation.

## Replicating Paper Results

The data for the mainline S5 run is stored in the `paper` directory itself, and corresponds to `batch_experiments/S5_MLP_seed2`. The rest of the data is sourced from the `batch_experiments` directory. To produce the evidence as in the paper, navigate to the "paper" directory, and run the notebook `mainline_results_and_plots.ipynb` (mostly for figures) and script `tables.py` (mostly for tables). 

The source of this data was produced by running `create_run_and_eval_batch_experiments.sh`. This script first creates all the experiments, then runs them, and finally evaluate the results, by iterating over every checkpoint and calculating various metrics. The final models, checkpoints, data, and figures output is stored in the `batch_experiments` directory. The code will cache various files to `utils/cache`, containing large tensors containing group multiplication tables, and large tensors dependent on group representations. To aid replication and further work, we provide these large cached files on google drive [here](https://drive.google.com/file/d/13qzbFHULsKiV77lII6CciiCLU3ErTExC/view?usp=sharing). A script that downloads them and extracts them to the correct location is provided in `utils/cache/download_and_extract_cache.sh`.

The checkpoints are not needed to produce many of the paper results. We provide the final models, summary statistics, and figures in the `batch_experiments` directory, only omiting the checkpoints. By modifying the contents of `batch_experiments\experiments_to_eval` and running `eval_batch_experiments_final.sh`, one should be able to reevaluate many final models. One can write their own metrics to evaluate by modifying the file `utils\metrics.py`.  

