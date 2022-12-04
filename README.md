## Overview

Computational models are powerful tools that can enable the optimization of deep brain stimulation (DBS). To enhance the clinical practicality of these models, their computational expense and required technical expertise must be minimized. An important aspect of DBS models is the prediction of neural activation in response to electrical stimulation. Existing rapid predictors of activation simplify implementation and reduce prediction runtime, but at the expense of accuracy. We sought to address this issue by leveraging the speed and generalization abilities of artificial neural networks (ANNs) to create a novel predictor of neural fiber activation in response to DBS. An accurate and fast ANN-based predictor would enable greater utilization of computational DBS models within clinical settings, and as a result increase the efficacy of DBS treatments.

![overview](https://github.com/jgolabek/ANN_Rapid_Predictor/blob/main/images/overview.PNG)

## Code

#### ANN
Code for creating ANN-based predictor models. ANN models are created using [TensorFlow](https://www.tensorflow.org/). Models are trained and optimized/validated with datasets generated from simulations of the MRG axon model. The optimal hyperparameters are identified through a random search that is parallelized on [HiPerGator](https://www.rc.ufl.edu/about/hipergator/).

#### MRG
Code for simulating multicompartment models of axons, using the [McIntyre, Richardson, and Grill (MRG) double-cable model](https://journals.physiology.org/doi/full/10.1152/jn.00353.2001). This is the current "gold standard" for estimating the neural response to electrical stimulation. MRG simulations are performed in parallel on HiPerGator.

#### lib
Contains python libraries that implement core components required for developing predictor models. Currently consists of three groups:  (1) finite element model (FEM) data processing in [COMSOL](https://www.comsol.com/), (2) diffusion tensor imaging (DTI) based tractography data processing in python, and (3) MRG axon simulation in [NEURON](https://neuron.yale.edu/neuron/). 

#### graphing
Code to graph various data of interest, including ANN prediction error and optimization.

#### debug
Various scripts to aid in debugging and experimentation.

#### misc
Miscellaneous scripts to test ideas, create figures, etc.

####
Note: Certain large data files, such as the csv datasets used to train the ANNs, are not included in this repository due to github size constraints