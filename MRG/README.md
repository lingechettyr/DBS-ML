## Overview

Code for simulating multicompartment models of axons, using the [McIntyre, Richardson, and Grill (MRG) double-cable model](https://journals.physiology.org/doi/full/10.1152/jn.00353.2001). This is the current "gold standard" for estimating the neural response to electrical stimulation. In this project, various methods are used to determine the trajectory of axons to simulate: (1) straight axons, orthogonal to the DBS lead, (2) realistic trajectories obtained from group-averaged DTI-based tractography, and (3) artificially non-straight trajectories generated from an iterative algorithm and spline interpolation.

## Guide

Determine which parameters and ranges to simulate. Modify [create_param_grid.py](create_param_grid.py) accordingly. Run with the desired output csv filename (put in params_LUT/) as a command line argument:

```bash
python create_param_grid.py <hpg_params.csv>
```

If simulating straight fibers:
Modify [straight_axon_multi.py](straight_axon_multi.py) with the desired parameters (number of nodes per axon, diameter, orientation, etc.). Modify [hpg_straight_axon_multi.sh](hpg_straight_axon_multi.sh) with the correct array size (number of rows in hpg_params.csv). Submit the job to the HPG scheduler while specifying the FEM solution file, output directory (where to store the json files), parameter csv file, offset (only nonzero if multiple submissions required because array size is greater than 3000), and whether the fibers' orientation to the DBS lead (0 for orthogonal, 1 for parallel):

```bash
sbatch hpg_straight_axon_multi.sh <comsol_export_file.txt> <json_results_dir> <hpg_params.csv> <offset> <orientation>
```

If simulating fibers obtained from DTI-based tractography or the iterative algorithm:
Modify [dti_axon_multi.py](dti_axon_multi.py) with the desired parameters. Modify [hpg_dti_axon_multi.sh](hpg_dti_axon_multi.sh) with the correct array size. Submit the job to the HPG scheduler while specifying the FEM solution file, axon trajectory file, output directory, parameter csv file, and offset:

```bash
sbatch hpg_dti_axon_multi.sh <comsol_export_file.txt> <tract_file.txt> <json_results_dir> <hpg_params.csv> <offset>
``` 

To process the results of the above simulations and output them as a json look-up-table (LUT), run [create_result_LUT.py](create_result_LUT.py) with the input json directory, output file location, and whether the results from straight fiber sims or DTI fiber sims are being processed.

To simulate the MRG model for DTI axons locally, use the [dti_axon_local.py](dti_axon_local.py) script. The FEM solution and tracts file are specified on the command line, and the results are graphed using the mayavi mlab library.

To find the activation initiation sites of MRG axons locally, use the [dti_axon_init_site_finder.py](dti_axon_init_site_finder.py) script. Jsons giving the simulation results, as well as images showing the initiation sites, are generated and saved for each axon.