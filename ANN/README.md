## Overview

Code for creating and using ANN-based predictor models. ANN models are created via [TensorFlow](https://www.tensorflow.org/). Models are trained and validated with datasets generated from simulations of the MRG axon model. The optimal hyperparameters are identified through a random search that is parallelized on [HiPerGator](https://www.rc.ufl.edu/about/hipergator/). Two types of models can be created: (1) Classification models, where the output is a prediction of axonal activation, or (2) Regression models, where the output is a prediction of stimulus activation threshold. Models can be trained on MRG simulations of straight orthogonal axons or (artificially) tortuous axons.

![architecture](https://github.com/jgolabek/ANN_Rapid_Predictor/blob/main/images/model_architecture.png)

## Guide

Modify [create_dataset.py](create_dataset.py) (or [create_orth_dataset.py](create_orth_dataset.py)) with the desired number and types of datasets for use by an ANN. Modify [hpg_create_dataset.sh](hpg_create_dataset.sh) (or [hpg_create_orth_dataset.sh](hpg_create_orth_dataset.sh)) with the correct array size (number of datasets). Submit the job to HPG while specifying the path to the directory containing the json files (MRG simulation results) and the directory to which to store the created csv datasets:

```bash
sbatch hpg_create_dataset.sh <json_data_dir> <csv_dataset_dir>
```
or
```bash
sbatch hpg_create_orth_dataset.sh <json_data_dir> <csv_training_dataset_dir> <csv_validation_dataset_dir>
```

Modify [create_hparam_set.py](create_hparam_set.py) with the desired hyperparameter types and ranges to search over. Run the script with the output csv path (params_LUT/ directory) and size of the search as command line arguments:

```bash
python create_hparam_set.py <csv_output_file> <search_size>
```

Modify [hparam_search_multi.py](hparam_search_multi.py) with the location of the datasets, as well as the activation functions. Load the tensorflow/2.4.1 module in HPG and run the search. (The offset input for hpg_hparam_search_multi should be 0 unless performing a search with size > 3000. HPG limits array jobs to size 3000 and so an offset would be used to split a large search into smaller ones).

```bash
module load tensorflow/2.7.0
```
and
```bash
sbatch hpg_hparam_search_multi.sh <hparam_csv> <offset>
```

The results of the search will be stored to a ''runs/" directory. Download and then view these results in [TensorBoard](https://www.tensorflow.org/tensorboard):

```bash
tensorboard --logdir <runs>
``` 

Identify the optimal hyperparameter combination, and modify [ann_train.py](ann_train.py) with the these hyperparameter values and the desired datasets with which to train/evaluate. Run while specifying the desired name of the model (should store in the [saved_models](saved_models) directory along with the normalization parameters):

```bash
python ann_train.py <model_name>
```

[ann_predict_lib.py](ann_predict_lib.py) provides a class with functions that make using the ANNs easier. 

To use an ANN to predict the thresholds of DTI-based fibers and store them to a json LUT, use [dti_ann_LUT.py](dti_ann_LUT.py). To make binary-activation predictions and graph the results, use [dti_ann_graph.py](dti_ann_graph.py).

To make ANN predictions on straight fibers and store to a LUT, instead use the [straight_ann_LUT.py](straight_ann_LUT.py) script.