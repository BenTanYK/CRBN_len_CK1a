import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import dist
from tqdm import tqdm


def obtain_RMSD(run_number, res_range=[0,606], eq=False):
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')
    protein = u.select_atoms("protein")

    if eq==True:
        u_ref = u = mda.Universe('equilibrated_structures/complex_eq.prmtop', 'equilibrated_structures/complex_eq.inpcrd')
        ref = u_ref.select_atoms("protein")
    
    else:
        ref = protein

    R_u =rms.RMSD(protein, ref, select=f'backbone and resid {res_range[0]}-{res_range[1]}')
    R_u.run()

    rmsd_u = R_u.rmsd.T #take transpose
    time = rmsd_u[1]/1000
    rmsd= rmsd_u[2]

    return time, rmsd

def save_RMSD(run_number, res_range=[0,606]):
    """
    Save the RMSD of a given run in a .csv file
    """
    time, RMSD = obtain_RMSD(run_number, res_range)

    df = pd.DataFrame()
    df['Time (ns)'] = time
    df['RMSD (Angstrom)'] = RMSD

    filename = 'RMSD.csv'

    df.to_csv(f"results/run{run_number}/{filename}")

    return df

def obtain_RMSF(run_number, res_range=[0,606]):
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')
    
    start, end = int(res_range[0]), int(res_range[1])

    alignment_selection = f'protein and name CA and resid {start}-{end}'
    c_alphas = u.select_atoms(alignment_selection)
    if len(c_alphas) == 0:
        raise ValueError(f"No atoms selected with selection: '{alignment_selection}'")

    # build average structure 
    avg = align.AverageStructure(u, select=alignment_selection, ref_frame=0)
    avg.run()
    ref = avg.results.universe

    # align trajectory in memory 
    align.AlignTraj(u, ref, select=alignment_selection, in_memory=True).run()

    # compute RMSF
    R = rms.RMSF(c_alphas)
    R.run()

    return c_alphas.resids, R.results.rmsf

def save_RMSF(run_number, res_range=[0,606]):
    """
    Save the RMSD of a given run in a .csv file
    """
    residx, RMSF = obtain_RMSF(run_number, res_range)

    df = pd.DataFrame()
    df['Residue index'] = residx
    df['RMSF (Angstrom)'] = RMSF
    df.to_csv(f"results/run{run_number}/RMSF.csv")

    return df

def run_analysis(systems, k_values):

    for system in systems:
        for k_DDB1 in k_values:
            for n_run in [1,2,3]:
                print(f"\nGenerating RMSD 1for {system}, run {n_run}, with k={k_DDB1} kcal/mol AA^-2\n")
                save_RMSD(system, k_DDB1, n_run, glob=True)
                save_RMSD(system, k_DDB1, n_run, glob=False)
                print(f"\nGenerating RMSF for {system}, run {n_run}, with k={k_DDB1} kcal/mol AA^-2\n")
                save_RMSF(system, k_DDB1, n_run)

for n_run in [1,2,3]:
    # complex
    print(f"\nGenerating RMSD for run {n_run}")
    save_RMSD(n_run)

    # Complex wrt equilibrated structure
    # time, RMSD = obtain_RMSD(n_run, eq=True)
    # df = pd.DataFrame()
    # df['Time (ns)'] = time
    # df['RMSD (Angstrom)'] = RMSD
    # filename = 'RMSD_eq_ref.csv'
    # df.to_csv(f"results/run{n_run}/{filename}")

    # CRBN
    time, RMSD = obtain_RMSD(n_run, [0,312])
    df = pd.DataFrame()
    df['Time (ns)'] = time
    df['RMSD (Angstrom)'] = RMSD
    filename = 'RMSD_CRBN.csv'
    df.to_csv(f"results/run{n_run}/{filename}")

    # CK1a
    time, RMSD = obtain_RMSD(n_run, [314, 606])
    df = pd.DataFrame()
    df['Time (ns)'] = time
    df['RMSD (Angstrom)'] = RMSD
    filename = 'RMSD_CK1a.csv'
    df.to_csv(f"results/run{n_run}/{filename}")


    # complex
    print(f"\nGenerating RMSF for  run {n_run}")
    save_RMSF(n_run)   

    # CRBN
    residx, RMSF = obtain_RMSF(n_run, [0,312])
    df = pd.DataFrame()
    df['Residue index'] = residx
    df['RMSF (Angstrom)'] = RMSF
    df.to_csv(f"results/run{n_run}/RMSF_CRBN.csv")

    # CK1a
    residx, RMSF = obtain_RMSF(n_run, [314,606])
    df = pd.DataFrame()
    df['Residue index'] = residx
    df['RMSF (Angstrom)'] = RMSF
    df.to_csv(f"results/run{n_run}/RMSF_CK1a.csv") 

