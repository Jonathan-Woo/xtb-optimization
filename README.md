# XTB Optimization

## Setup

Download the reference VQM24 dataset from [zenodo]{https://zenodo.org/records/11164951}.

Create environment with `conda env create -f environment.yml`

## Run

xtb is run from the command line where we pass it a .xyz and configuration file. For my experiments, I've split the work into two scripts,

I've used multiprocessing & multithreading heavily in this so please feel free to configure the pool sizes.

1. `setup_experiment.py`
   1. Samples a set of molecules from VQM24.
   2. Generates a json of parameters to test.
   3. Generates xtb configuration files for those parameters in 2.
2. `experiment.py`
   - Executes xtb for each experiment setup in the previous step.
3. `results_processing.ipynb`
   1. Parses the xtb output files to collect atomisation energy results.
   2. Retrieves associated VQM24 atomisation energies.
   3. Calculates MAE
   4. Plots
