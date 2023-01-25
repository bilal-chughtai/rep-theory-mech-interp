# Associated Code for Paper "A Toy Model of Universality: Reverse Engineering how Networks Learn Group Operations"

## Setup

To run code, create a virtual environment and install the requirements listed under `requirements.txt`. The code was tested with Python 3.8.10

## Replicating Paper Results

To reproduce the results in the paper, run "create_run_and_eval_batch_experiments.sh". This will first create the batch experiments, then run them, and finally evaluate the results. The final model, checkpoints, data, and figures will be stored in the "batch_experiments" directory. The code will cache various files to "utils/cache", containing large tensors we use toe evaluate models.

To produce the evidence as in the paper, navigate to the "paper" directory, and run the notebook and script. The cached "mainline-S5" model is stored in the "paper" directory, and corresponds to "batch_experiments/S5_MLP_seed2" 
