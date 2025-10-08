import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import dist
import pickle
from tqdm import tqdm
import os

def getDistance(idx1, idx2, u):
    """
    Get the distance between two atoms in a universe.

    Parameters
    ----------
    idx1 : int
        Index of the first atom
    idx2 : int
        Index of the second atom
    u : MDAnalysis.Universe
        The MDA universe containing the atoms and
        trajectory.

    Returns
    -------
    distance : float
        The distance between the two atoms in Angstroms.
    """
    distance = dist(
        mda.AtomGroup([u.atoms[idx1]]),
        mda.AtomGroup([u.atoms[idx2]]),
        box=u.dimensions,
    )[2][0]
    return distance

def closest_residue_to_point(atoms, point):
    """Find the closest residue in a selection of atoms to a given point"""
    residues = atoms.residues
    distances = np.array([np.linalg.norm(res.atoms.center_of_mass() - point) for res in residues])

    # Find the index of the smallest distance
    closest_residue_index = np.argmin(distances)

    # Return the closest residue
    return residues[closest_residue_index], distances[closest_residue_index]

def obtain_CA_idx(u, res_idx):
    """Function to obtain the index of the alpha carbon for a given residue index"""
    
    selection_str = f"protein and resid {res_idx} and name CA"
    
    selected_CA = u.select_atoms(selection_str)

    if len(selected_CA.indices) == 0:
        print('CA not found for the specified residue...')
    
    elif len(selected_CA.indices) > 1:
        print('Multiple CAs found, uh oh...')

    else:  
        return selected_CA.indices[0]
    
def obtain_angle(pos1, pos2, pos3):

    return mda.lib.distances.calc_angles(pos1, pos2, pos3)

def obtain_dihedral(pos1, pos2, pos3, pos4):
   
    return mda.lib.distances.calc_dihedrals(pos1, pos2, pos3, pos4)

def obtain_Boresch_dof(dof, u, rec_group, lig_group, res_b, res_c, res_B, res_C):

    """
    Calculate a Boresch DOF (thetaA, thetaB, phiA, phiB, phiC) 
    Specify the recepter and ligand interface as rec_group and lig_group
    The other anchor points are given as res_b, res_c, res_B and res_C
    """

    group_a = u.atoms[rec_group]
    group_b = u.atoms[[obtain_CA_idx(u, res_b)]]
    group_c = u.atoms[[obtain_CA_idx(u, res_c)]]
    group_A = u.atoms[lig_group]
    group_B = u.atoms[[obtain_CA_idx(u, res_B)]]
    group_C = u.atoms[[obtain_CA_idx(u, res_C)]]

    pos_a = group_a.center_of_mass()
    pos_b = group_b.center_of_mass()
    pos_c = group_c.center_of_mass()
    pos_A = group_A.center_of_mass()
    pos_B = group_B.center_of_mass()
    pos_C = group_C.center_of_mass()

    dof_indices = {
        'thetaA' : [pos_b, pos_a, pos_A],
        'thetaB' : [pos_a, pos_A, pos_B],
        'phiA' : [pos_c, pos_b, pos_a, pos_A],
        'phiB': [pos_b, pos_a, pos_A, pos_B],
        'phiC': [pos_a, pos_A, pos_B, pos_C]
    }

    indices = dof_indices[dof]

    if len(indices) == 3:
        return obtain_angle(indices[0], indices[1], indices[2])

    else:
        return obtain_dihedral(indices[0], indices[1], indices[2], indices[3])


# Specify interface residues and anchor points
rec_group =  [545, 562, 578, 1227, 1243, 3800, 3816, 3838, 3844, 3861, 3868, 3889, 3905, 4102, 4116, 4135, 4154, 4161, 4193, 4199, 4210, 4224, 4239, 4256, 4363, 4387, 4401, 4417, 4427, 4454, 4476, 4495, 4505, 4515, 4526, 4543, 4562, 4569, 4880, 9886, 9887, 9888, 9889, 9890, 9891, 9892, 9893, 9894, 9895, 9896, 9897, 9898, 9899, 9900, 9901, 9902, 9903, 9904]
lig_group =  [5078, 5100, 5121, 5358, 5377, 5387, 5406, 5420, 5439, 5453, 5467, 5474, 5489, 5504, 5520, 5941, 5958, 6028, 6045, 6064, 6292, 6309, 6321, 6340, 6359, 7215, 7222, 7241, 7248, 7272]

res_b = 90
res_c = 172 
res_B = 424 
res_C = 508 

for run_number in [1,2,3]:

    for dof in ['thetaA', 'thetaB', 'phiA', 'phiB', 'phiC']:

        if os.path.exists(f'results/run{run_number}/{dof}.pkl'):
            continue
        else:
            print(f"Performing Boresch analysis for {dof} run {run_number}")

            u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')

            vals = []

            for ts in tqdm(u.trajectory[:-500], total=(u.trajectory.n_frames-500), desc='Frames analysed'):
                vals.append(obtain_Boresch_dof(dof, u, rec_group, lig_group, res_b, res_c, res_B, res_C))

            frames = np.arange(1, len(vals) + 1)

            dof_data = {
                'Frames': frames,
                'Time (ns)': np.round(0.01 * frames, 6),
                'DOF values': vals
            }

            # Save interface data to pickle
            file = f'results/run{run_number}/{dof}.pkl'
            with open(file, 'wb') as f:
                pickle.dump(dof_data, f)