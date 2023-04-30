# Associated Code for Paper "A Toy Model of Universality: Reverse Engineering how Networks Learn Group Operations"

## Setup

To run the code, create a virtual environment and install the requirements listed under `requirements.txt`. The code was tested with Python 3.8.10.

## Replicating Paper Results

To reproduce the tables and figures in the paper, navigate to the "paper" directory, and run the notebook and script. 

The data for the mainline S5 run is stored in the `paper` directory itself, and corresponds to `batch_experiments/S5_MLP_seed2`. The rest of the data is sourced from the `batch_experiments` directory. To produce the evidence as in the paper, navigate to the "paper" directory, and run the notebook (mostly for figures) and script (mostly for tables). 

This data was produced by running `create_run_and_eval_batch_experiments.sh`. This script first creates all the experiments, then runs them, and finally evaluate the results, by iterating over every checkpoint and calculating various metrics. The final models, checkpoints, data, and figures output is stored in the `batch_experiments` directory. The code will cache various files to `utils/cache`, containing large tensors containing group multiplication tables, and large tensors dependent on group representations.

This script takes a long time to run. To aid replication and further work, we provide these large cached files on google drive [HERE]. A script that downloads them and extracts them to the correct location is provided in `utils/cache/download_and_extract_cache.sh`. TODO. We also provide the final models, summary statistics, and figures in the `batch_experiments` directory, only omiting the checkpoints. The checkpoints are available upon request.

