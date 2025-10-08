# CK1a-CRBN binding free energy

This repository contains a collection of scripts for calculating the binding free energy of the CK1a-CRBN complex. The methodology is based on the geometric route of [Woo and Roux ](https://www.pnas.org/doi/10.1073/pnas.0409005102), with [Boresch-style restraints](https://pubs.acs.org/doi/10.1021/jp0217839) being used to restrain the relative orientation of the two proteins, as per the framework of [Notari *et al*](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01695). Umbrella Sampling (US) simulations are used to calculate the Potential of Mean Force (PMF) curves corresponding to each stage of the binding/unbinding process. The MD engine used for all simulations is [OpenMM](https://openmm.org/).

The protocol follows three main stages: 

1. **Unrestrained MD** for obtaining equilibrated starting structures and equilibrium values for the various Boresch DOFs
2. **Steered MD** for generating input configurations for the separation 
3. **Umbrella Sampling** simulations for calculating the various free energy contributions required for calculating the overall CK1a-CRBN binding free energy

## Requirements for running the protocol

- PyMol for editing and generating input structures
- [Ambertools](https://anaconda.org/conda-forge/ambertools) for generating input topologies and coordinates
- Python environment with openmm<8.3, pandas and MDAnalysis

## Running unrestrained MD

1. Prepare your .prmtop/.inpcrd input files within the ```UnrestrainedMD/structures``` directory,  e.g. using ```tleap -f tleap.in```.
2. Submit an MD simulation to slurm using ```sbatch submitMD.sh -r $RUN_NUMBER```.
3. Once three MD simulations have finished, calculate the RMSD and RMSF for the trajectories using ```analyse_traj.py```.
4. Identify the interface residues and anchor points using the ```MD_analysis.ipynb``` notebook.
5. Sample the Boresch Degrees of Freedom (DOFs) during the MD trajectories using ```analyse_traj.py```.
6. Extract the equilibrated input structures and the equilibrium values for the Boresch DOFs (using MD_analysis.ipynb).

## Running Steered MD

The SMD simulation is performed in the ```umbrella_sampling/separation``` directory.
Ensure that you have copied the input files for the equilibrated complex over to the ```structures``` directory, under the name ```complex_eq.prmtop/.inpcrd```. 

Modify the following parameters in ```SMD.py``` to match your system: 

- Residues indices for selecting the atoms restrained by the RMSD potential
- Interface residues for CRBN and the target
- Anchor point residue indices and equilibrium values for the Boresch DOFs

Once you're satisfied with the SMD setup, submit the simulation using ```python SMD.py```

## Submitting jobs to slurm

The ```scripts``` directory contains slurm/grid engine submission scripts for the US simulations. Modify ```submit_window_workstation.sh``` to match the slurm configuration on your local workstation. 

## Generating the separation PMF

An Umbrella Sampling simulation can be submitted using the ```submitrun``` executable. Each MD simulation is performed by the ```run_window.py``` script, which must be again modified to account for the correct indices and equilibrium values in the RMSD and Boresch restraints. 

Before submitting the ```submitrun``` executable, modify the params.in file to specify all parameters (force constants, sampling time, etc.) You can specify the window spacing by providing comma separated values in the r0_vals.list file. 

## RMSD and Boresch US

The ```run_window.py``` file must be again suitably modified within ```umbrella_sampling/RMSD``` and ```umbrella_sampling/Boresch``` to match the system of interest. Simulations are submitted via preparation of ```params.in``` and ```CV0_vals.list``` in the same way as for the separation PMF. 

## Analysis

```umbrella_sampling/analysis.py``` contains useful functions for generating PMFs using WHAM, analysing the individual US windows and calculating the corresponding free energy contributions for the various stages of the thermodynamic cycle.

The notebook ```umbrella_sampling/analysis.ipynb``` contains some useful analysis code, including visualisation of average PMFs and automated calculation of free energy contributions.


