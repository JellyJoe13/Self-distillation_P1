# Self-distillation_P1

## Project introduction
Self distillation is a technique which works together with Neural Networks(Deep Learning). It is an addition where 
unlabelled data is fed back into the training loop after predicting the label with the pre-trained model.

This project is meant to find out if self distillation is beneficial in the context of predicting protein ligand
interactions when using Random Forests and Graph Neural Networks (shallow).

This project is the result of the university course "Praktikum 1" of master studies of Computer Science 
(https://ufind.univie.ac.at/de/course.html?lv=053021&semester=2023S).


Author: Johannes P. Urban, B.Sc.

Supervisor: Ass.-Prof. Dipl.-Inf. Dr. Nils Morten Kriege

Co-supervisor: Steffen Hirte, B.Sc. M.Sc. M.Sc.

The report and hence the results of this project can be read in the pdf report 
'P1_Urban_Self-Distillation-on-Graph-Neural-Networks-for-Molecule-Property-Prediction.pdf'

## Content description
- data (see more in data section)
- docs - html documentation hosted via GitHub pages on https://jellyjoe13.github.io/Self-distillation_P1/
- docs-config - setup and rst files for documentation
- notebooks - keeps the majority of experiment notebooks for GNN related experiments
- results - storage of results produced by jupyter notebooks
- selfdist_toolkit - contains python library/routines to make experimental notebooks simpler to read and make techniques reusable
  - data_tools - routines aimed to do data operations/preprocessing
  - pyg_tools - tools related to GNN experiments
  - randomforest_tools - tools related to RF experiments 
- environment(2).yml - information file on environment setup - not directly usable to reproduce to environment because of pytorch setup 
- installation-procedures.txt - concise textual guide to setup environment on test machine - alter for own machine using 
  [pytorch documentation](https://pytorch.org/get-started/locally/) and [pytorch geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- P1_Urban_Self-Distillation-on-Graph-Neural-Networks-for-Molecule-Prediction.pdf - report of this project
- requirements(2).txt - list of python libraries used and their version - not usable for environment reproduction
- notebooks at top level - these are the notebooks for development, Random Forest experiments and the first GNN tests.
Because of file structure it is kept at this level because else path references need to be changed which requires
testing of the changed paths and hence recomputation of results which cascades quickly because the main teacher model
if affected.

## Data fetching
Use the ucloud repository via [link](https://ucloud.univie.ac.at/index.php/s/XcnZ8q13sqQgraT) to download either whole 
data folder or only the dataset itself:

### Data folder (recommended)
Download and unpack data.zip in the top folder Self-distillation_P1, make sure that there is not a nested 'data' folder
in the data folder.

### Dataset only
(In case the compressed df_assay_entries.zip is downloaded - decompress it first).
Create a folder 'data' and place the file 'df_assay_entries.csv' into this folder. The experiments of this project 
require preprocessed data (RF needs chemical descriptor data and morgan fingerprint (precomputed to save time later on)
and GNN needs pre-split data so that not whole dataset is loaded for one chemical experiment but only a smaller part of
the dataset).

To preprocess data call the function 'experiment_whole_preprocess' from the library path 
'selfdist_toolkit.data_tools.preprocessing' using the following python commands:
```python
from selfdist_toolkit.data_tools import preprocessing

# path to dataset
PATH_DATA = "data/"
PATH_MAIN_DATASET = PATH_DATA + "df_assay_entries.csv"

preprocessing.experiment_whole_preprocess(PATH_MAIN_DATASET, PATH_DATA)
```
This python code can also be found in the notebook 'Random_Forest.ipynb'. It will check if the dataset is already
downloaded (if not it will attempt a download from ucloud which does not always work - please make sure the dataset is
already downloaded). It will then split the dataset and compute the chemical descriptor data and morgan fingerprint data
for the Random Forest tests. This will use all cores your system has to offer and depending on your systems performance
may take up to approximately 2-3 hours in the worst case = time for one processor.

If your internet connection is good enough it is recommended to download the already pre-computed zip and unpack it to 
save time, as computing this will stall your PC for some time.

## Installation guideline
For detailed information see file installation-procedures.txt, here is a sketch of the rough installation process
- Create conda environment (with python 3.9) - e.g. ```conda create --name <xyz> python=3.9.*```
- Activate environment - ```conda activate <xyz>```
- Install torch if possible with CUDA - see [link](https://pytorch.org/get-started/locally/) - e.g. ```pip3 install torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117```
- Install pytorch geometric - see [link](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) - e.g. ```pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html```
- Install basic libraries numpy, pandas, rdkit, scikit-learn
  - ```pip install numpy pandas rdkit scikit-learn```
  - ```conda install numpy pandas rdkit scikit-learn```
- Install advanded libraries multiprocess(parallelization), jupyterlab(notebooks), matplotlib(plotting)
  - ```pip install multiprocess jupyterlab matplotlib```
  - ```conda install multiprocess jupyterlab matplotlib```

In the case you are attempting to reproduce the results and are unable to do so because of library changes, please refer
use the versions of the libraries specified in the requirements.txt.

I apologize for the inconvenience however this steps are required as - as stated earlier - using either the
environment.yml or the requirements.txt to reproduce the environment does not work - most likely because of the special
torch CUDA setup.