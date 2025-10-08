"""
Analysis functions for US simulations
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import red
import shutil
import warnings
import subprocess
from tqdm import tqdm

"""Defining constants"""

temperature = 298.15 # K
boltzmann = 0.0019872041 # kcal/mol K
beta = 1.0/(boltzmann*temperature)
k_boresch = 100 # kcal/mol rad**2
k_rmsd = 3000 # kcal/mol nm**-2
sep_cv_max = 3.3 # nm
standard_volume = 1660 # angstroms^3
standard_volume_nm = standard_volume*0.001 # nm^3
radius_sphere = (3*standard_volume_nm/(4*np.pi))**(1.0/3.0) # radius of sphere whose volume is equal to the standard volume in nm

def obtain_dirpath(free_energy_step, restraint_type, dof, equilibration, sampling_time=None, run_number=1):
    """
    Helper function to obtain the results directory for a given set of results

    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    restraint_type : str
        Type of RMSD restraint applied to the system
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)     
    sampling_time : int, Optional, default=None
        Length of sample, if None then use the full set of CVs     
    run_number : int, default=1
        Specify the replicate

    Returns
    -------
    dirpath : str
        Directory for desired set of results
    """
    
    dirpath = f"{os.getcwd()}/{free_energy_step}/results/{restraint_type}_RMSD"
    # dirpath = f"{os.getcwd()}/{free_energy_step}/results/{restraint_type}_RMSD"

    if free_energy_step in ('Boresch', 'RMSD'):
        dirpath+=f"/{dof}"

    dirpath+=f"/run{run_number}/"

    if equilibration == None:
        pass
    
    elif equilibration == 'RED':
        dirpath+=f"RED"

    elif isinstance(equilibration, int):
        dirpath+=f"{equilibration}ns_equil"

    # Specify sample size
    if isinstance(sampling_time, int):
        dirpath+=f"_{sampling_time}ns_sampling"


    return dirpath

def obtain_sampling_time(free_energy_step, restraint_type, dof, run_number):
    """
    Obtain the total sampling time for the full set of CV samples.
    This assumes that all files for a given run have the same sampling."""
    dirpath = obtain_dirpath(free_energy_step, restraint_type, dof, equilibration=None, sampling_time=None, run_number=run_number)
    filenames = [file for file in os.listdir(dirpath) if file.endswith('.txt')]
    
    # Filter any text files that don't contain CV samples
    CV_samples = []

    for filename in filenames: 
        parts = filename.split('.txt')

        try:
            CV =  float(parts[0])
            CV_samples.append(filename)

        except:
            pass

    sample_file = dirpath + '/' + filenames[0]

    # Assume we have 0.5 ps sampling interval
    n_samples = len(np.loadtxt(sample_file))

    return 0.5E-3*n_samples

def truncate_data(truncation_time, free_energy_step, restraint_type, sampling_time=None, dof=None, run_number=1):
    """
    Generate a new set of truncated CV files for a given run
    Parameters
    ----------
    truncation_time : int
        Time in ns to cut at the start of each CV file
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
        Options are:
        - "separation" : radial BD2-DCAF16 separation
        - "RMSD" : applying/removing conformational restraints
        - "Boresch" : applying orientational restraints in the bound state
    restraint_type : str
        Type of RMSD restraint applied to the system
        Options are: 
        - heavy_atom
        - backbone
        - CA
    sampling_time : int or None
        Sampling time in ns (after equilibration). The default is None - this will use
        the full 30 ns of sampling (with truncated data removed)
    dof : str
        Degree of freedom for which to generate metafile
        The default option is:
        - None - this is used for the radial separation, where there is a only a single dof
        Options for free_energy_step = "RMSD":
        - BD2 (unbound)
        - DCAF16 (unbound)
        - BD2_only (bound)
        - DCAF16_only (bound)
        - BD2withDCAF16 (bound)
        - DCAF16withBD2 (bound)
        Options for free_energy_step = "Boresch":
        - thetaA
        - thetaB
        - phiA
        - phiB
        - phi
    run_number : int, default=1
        Specify the replicate
    plot : bool, default = True
        Generate CV histogram

    Returns
    -------
    None
    """  
    # Assign the sampling time
    try:
        total_time = obtain_sampling_time(free_energy_step, restraint_type, dof, run_number)
    except:
        raise ValueError('Select one of the following free energy steps: "separation", "RMSD", "Boresch"!')

    # Check that sampling time does not exeed the total time
    if isinstance(truncation_time, int):
        if (sampling_time+truncation_time)>total_time:
            raise ValueError('Desired sampling time exceeds total measurement time...')

    #Directory path  
    dirpath = obtain_dirpath(free_energy_step, restraint_type, dof, None, None, run_number)

    savedir = f"{truncation_time}ns_equil"

    if isinstance(sampling_time, int):
        savedir+=f"_{sampling_time}ns_sampling"

    if not os.path.exists(f"{dirpath}/{savedir}"):
        os.makedirs(f"{dirpath}/{savedir}")

    values = np.append(np.arange(-10, 0, 0.05), np.arange(0.0, 40.0, 0.05))

    # Identify CV values 
    CV_values = []
    for value in values: 
        value = np.round(value, 3)
        inputname = f"{dirpath}/{value}.txt"
        if os.path.exists(inputname):
            CV_values.append(value)

    # Copy over all files that haven't been truncated
    for CV_value in tqdm(CV_values, desc='Truncating files', total=len(CV_values)):
        CV_value = np.round(CV_value, 3)

        inputname = f"{dirpath}/{CV_value}.txt"
        outputname = f"{dirpath}/{savedir}/{CV_value}.txt"

        try: 
            if os.path.exists(outputname):
                pass
                # print(f'Truncated file already exists for {CV_value}')
            else:
                shutil.copy(inputname, outputname)

                # Truncate the copied file
                data = np.loadtxt(outputname)
                idx_start = int((truncation_time/total_time)*len(data))

                if sampling_time:
                    idx_stop = int(((sampling_time+truncation_time)/total_time)*len(data))
                else:
                    idx_stop = -1

                np.savetxt(outputname, data[idx_start:idx_stop])

                # print(f"Truncated datafile generated for {CV_value}")

        except:
            pass

def apply_RED(free_energy_step, restraint_type, sampling_time=None, dof=None, run_number=1, plot=False):
    try:
        # Assign the sampling time
        total_time = obtain_sampling_time(free_energy_step, restraint_type, dof, run_number)
    except Exception as e:
        raise ValueError('Select one of the following free energy steps: "separation", "RMSD", "Boresch"!') from e

    # Directory path  
    inputpath = obtain_dirpath(free_energy_step, restraint_type, dof, None, None, run_number)

    savedir = f"RED_{sampling_time}ns_sampling" if isinstance(sampling_time, int) else "RED"
    
    os.makedirs(f"{inputpath}/{savedir}", exist_ok=True)

    values = np.append(np.arange(-10, 0, 0.05), np.arange(0.0, 40.0, 0.05))

    # Identify CV values 
    CV_values = []
    for value in values:
        value = np.round(value, 3)
        inputname = f"{inputpath}/{value}.txt"
        if os.path.exists(inputname):
            CV_values.append(value)

    # Copy over all files that haven't been truncated
    for CV_value in tqdm(CV_values, desc='Applying RED', total=len(CV_values)):
        CV_value = np.round(CV_value, 3)
        inputname = f"{inputpath}/{CV_value}.txt"
        outputname = f"{inputpath}/{savedir}/{CV_value}.txt"

        try:
            if not os.path.exists(outputname):
                shutil.copy(inputname, outputname)

                # Truncate the copied file
                data = np.loadtxt(outputname)
                try:
                    # Subsample to provide RED with manageable amount of data
                    idx_start, g, ess = red.detect_equilibration_window(data[:20000:2, 1], method="min_sse", plot=plot)
                    idx_start*=2 # Multiply by 2 to account for subsampling
                except Exception as e:
                    print(f'Equilibration not detected in the first 10 ns, applying default 7 ns truncation for {CV_value}')
                    idx_start = int((3.5 / total_time) * len(data))

                # Check that sampling time does not exceed the total time
                if sampling_time:
                    if (sampling_time + (total_time * idx_start / len(data))) > total_time:
                        raise ValueError('Desired sampling time exceeds total measurement time...') 
                    idx_stop = int(((sampling_time + total_time * idx_start/len(data)) / total_time) * len(data))

                    # Check for correct sample length
                    if not (sampling_time * 2000 - 10 < (idx_stop-idx_start) < sampling_time * 2000 + 10):
                        raise ValueError(f'Incorrect sampling time, {idx_stop - idx_start} samples')
                                    
                else:
                    idx_stop = -1

                np.savetxt(outputname, data[idx_start:idx_stop]) 

        except Exception as e:
            print(f"Error processing {CV_value}: {e}")

def plot_timeseries(CV_value, free_energy_step, restraint_type, dof=None, equilibration=0, sampling_time=None, run_number=1):
    """
    Generate the metadata file required for WHAM implementation for a specific umbrella sampling run

    Parameters
    ----------
    CV_value : float
        Value of the CV of interest
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    restraint_type : str
        Type of RMSD restraint applied to the system
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)          
    run_number : int, default=1
        Specify the replicate
    plot : bool, default = True
        Generate CV histogram

    Returns
    -------
    None
    """
    plt.figure(figsize=(5,4), dpi=200)

    # Assign the sampling time
    try:
        total_time = obtain_sampling_time(free_energy_step, restraint_type, dof, run_number)
    except:
        raise ValueError('Select one of the following free energy steps: "separation", "RMSD", "Boresch"!')

    # Reading in files
    dirpath = obtain_dirpath(free_energy_step, restraint_type, dof, None, sampling_time, run_number)

    full_data = np.loadtxt(f"{dirpath}/{CV_value}.txt")[:,1]
    
    time = np.linspace(0, total_time, len(full_data))

    plt.plot(time, full_data, label='Full sampling')

    if equilibration == 'RED':
        RED_data = np.loadtxt(f"{dirpath}/RED/{CV_value}.txt")[:,1]
        RED_idx = len(full_data) - len(RED_data)
        plt.vlines(time[RED_idx], ymin=0.5*np.min(full_data), ymax=1.1*np.max(full_data), colors='r', linestyle='dashed', label='RED truncation')
    
    elif isinstance(equilibration, int) and equilibration>0:
        truncation_idx = int((equilibration/total_time)*len(full_data))
        plt.vlines(time[truncation_idx], ymin=0.5*np.min(full_data), ymax=1.1*np.max(full_data), colors='k', linestyle='dotted', label=f'{equilibration} ns truncation')
    
    plt.legend(loc='lower right')
    plt.xlabel('Time (ns)')
    plt.ylabel(f"{free_energy_step}")
    # plt.xlim(0,5)
    plt.show()

    return time, full_data
    
def generate_metafile(free_energy_step, restraint_type, dof=None, equilibration=None, sampling_time=None, run_number=1, force_constant=None, ignore_values=[], plot=True):
    """
    Generate the metadata file required for WHAM implementation for a specific umbrella sampling run

    Parameters
    ----------
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    restraint_type : str
        Type of RMSD restraint applied to the system
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)          
    run_number : int, default=1
        Specify the replicate
    force_constant : int, default=None
        Force constant (units of kcal mol-1/nm^2) used in US bias potentials, written to WHAM metafile
    ignore_values : list, default=[]
        List of CV values to ignore when writing to the metafile
    plot : bool, default = True
        Generate CV histogram

    Returns
    -------
    None
    """
    if plot == True:
        plt.figure(figsize=(5,4), dpi=200)

    if force_constant:
        k_CV = force_constant
    elif free_energy_step == 'Boresch':
        k_CV = 100 # 10 kcal mol-1/rad^2
    else:
        k_CV = 1000 # 10 kcal mol-1/Angstrom^2
    
    dirpath = obtain_dirpath(free_energy_step, restraint_type, dof, equilibration, sampling_time, run_number)
    
    if plot == True:
        print(dirpath)

    metafilelines = []

    # Perform equilibration if required
    if equilibration=='RED':
        apply_RED(free_energy_step, restraint_type, sampling_time, dof, run_number, plot=False)
    elif isinstance(equilibration,int):
        truncate_data(equilibration, free_energy_step, restraint_type, sampling_time, dof, run_number)

    values = np.append(np.arange(-10, 0, 0.05), np.arange(0.0, 40.0, 0.05))

    # Identify CV values 
    CV_values = []
    for value in values:
        value = np.round(value, 3)
        inputname = f"{dirpath}/{value}.txt"
        if os.path.exists(inputname) and value not in ignore_values:
            CV_values.append(value)

    for CV_value in CV_values: 
        CV_value = np.round(CV_value,4)
        filename = f"{CV_value}.txt"
        data = np.loadtxt(f'{dirpath}/{filename}')

        if free_energy_step == 'RMSD':
            metafileline = f'{dirpath}/{filename} {np.round(CV_value*0.1, 3)} {k_CV}\n'
        else:
            metafileline = f'{dirpath}/{filename} {CV_value} {k_CV}\n'

        if CV_value not in ignore_values:
            metafilelines.append(metafileline)                

        if plot==True:
            if free_energy_step != 'Boresch':
                plt.hist(10*data[:,1], bins=30, alpha=0.6, label=f"{CV_value}")
            else:
                plt.hist(data[:,1], bins=30, alpha=0.6, label=f"{CV_value}")


    if plot == True:
        plt.legend(fontsize='xx-small', ncol=6, loc='upper center', bbox_to_anchor=(0.45, -0.1))

    with open(f"{dirpath}/metafile.txt", "w") as f:    
        f.writelines(metafilelines)         

def perform_WHAM(wham_params, free_energy_step, restraint_type, dof=None, equilibration=0, sampling_time=None, run_number=1):
    """
    Perform WHAM to generate the PMF for a given US run

    Parameters
    ----------
    wham_params : list
        hist_min hist_max num_bins tol temperature numpad [num_MC_trials randSeed]
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    restraint_type : str
        Type of RMSD restraint applied to the system
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)          
    run_number : int, default=1
        Specify the replicate

    Returns
    -------
    None
    """
    dirpath = obtain_dirpath(free_energy_step, restraint_type, dof, equilibration, sampling_time, run_number)

    wham_path = '/home/btan/Software/WHAM/wham/wham/wham'

    if not os.path.exists(dirpath):
        raise FileNotFoundError(f"Directory does not exist: {dirpath}")

    if len(wham_params) == 6:
        wham_list = [wham_path] + wham_params + [f"{dirpath}/metafile.txt", f"{dirpath}/pmf.txt", f"> {dirpath}/wham.log"]
    elif len(wham_params) == 8:
        wham_list = [wham_path] + wham_params[:-2] + [f"{dirpath}/metafile.txt", f"{dirpath}/pmf.txt"] + wham_params[:-2] + [f"> {dirpath}/wham.log"]
    else:
        raise ValueError("Specify the following values in a list: hist_min hist_max num_bins tol temperature numpad [num_MC_trials randSeed]")

    wham_list = list(map(str, wham_list))

    command = ''
    for item in wham_list:
        command+=f"{str(item)} "

    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def obtain_PMF(free_energy_step, restraint_type, dof=None, equilibration=0, sampling_time=None, run_number=1, plot=True):
    """
    Generate the PMF plot for a specific US simulation
    Parameters
    ----------
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    restraint_type : str
        Type of RMSD restraint applied to the system
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)          
    run_number : int, default=1
        Specify the replicate


    Returns
    -------
    x : array
        CV values
    y : array
        PMF 
    """
    if plot == True:
        plt.figure(figsize=(5,4), dpi=200)

    dirpath = obtain_dirpath(free_energy_step, restraint_type, dof, equilibration, sampling_time, run_number)

    pmf = np.loadtxt(f'{dirpath}/pmf.txt')
    x=pmf[:,0]
    y=pmf[:,1]

    if plot == True:
        if free_energy_step != 'Boresch':
            plt.plot(10*x,y) #Units of Angstrom 
        else:
            plt.plot(x,y)
        plt.xlabel(f"{free_energy_step}")
        plt.ylabel('PMF (kcal/mol)')

    return x,y

def obtain_av_PMF(runs, free_energy_step, restraint_type, dof=None,
                  equilibration=0, sampling_time=None,
                  plot_indiv=False, plot_av=True):
    """
    Generate average PMF and corresponding errors

    Parameters
    ----------
    runs : list[int] or list[str]
        List of run indices (e.g. [1,2,3]) to combine into an average
    free_energy_step : str
        Section of the thermodynamic cycle where US simulation is performed
    restraint_type : str
        Type of RMSD restraint applied to the system
    dof : str, optional
        Degree of freedom for which to generate metafile (required if free_energy_step in ('Boresch','RMSD'))
    equilibration : 'RED' or int or 0, optional
        Type / amount of equilibration truncation ('RED' or integer number of ns). Default 0 (no equilibration folder)
    sampling_time : int or None, optional
        Sampling time in ns (appended to equilibration folder if specified)
    plot_indiv : bool, default=False
        Plot the individual PMFs for all replicas
    plot_av : bool, default=True
        Plot the average PMFs with the corresponding standard error

    Returns
    -------
    x : ndarray
        CV values (from pmf.txt first column)
    PMF_av : ndarray
        Average PMF (kcal/mol)
    PMF_err : ndarray
        Standard error in the mean PMF
    """
    # Basic validation
    if free_energy_step in ('Boresch', 'RMSD') and not dof:
        raise ValueError("dof must be provided when free_energy_step is 'Boresch' or 'RMSD'")

    resultspath = os.path.join(os.getcwd(), free_energy_step, 'results', f"{restraint_type}_RMSD")
    if free_energy_step in ('Boresch', 'RMSD'):
        resultspath = os.path.join(resultspath, dof)

    if not os.path.isdir(resultspath):
        raise FileNotFoundError(f"Result path does not exist: {resultspath}")

    pmfs = []
    x = None

    if plot_indiv:
        plt.figure(figsize=(6,4), dpi=200)
        plt.title(f'{free_energy_step} Individual PMFs')

    for n_run in runs:
        dirpath = os.path.join(resultspath, f"run{n_run}")

        # Build equilibration/sampling subfolder if requested
        if equilibration == 'RED':
            dirpath = os.path.join(dirpath, 'RED')
        elif isinstance(equilibration, int) and equilibration > 0:
            sub = f"{equilibration}ns_equil"
            if isinstance(sampling_time, int):
                sub += f"_{sampling_time}ns_sampling"
            dirpath = os.path.join(dirpath, sub)
        else:
            # no equilibration subfolder; if sampling_time alone is provided, optionally support it:
            if isinstance(sampling_time, int):
                sub = f"{sampling_time}ns_sampling"
                dirpath = os.path.join(dirpath, sub)

        pmf_file = os.path.join(dirpath, 'pmf.txt')
        if not os.path.isfile(pmf_file):
            warnings.warn(f"pmf.txt not found for run {n_run} at {pmf_file}; skipping this run.")
            continue

        try:
            pmf = np.loadtxt(pmf_file)
        except Exception as exc:
            warnings.warn(f"Failed to load {pmf_file} for run {n_run}: {exc}; skipping this run.")
            continue

        if pmf.ndim != 2 or pmf.shape[1] < 2:
            warnings.warn(f"Unexpected pmf.txt shape for run {n_run} ({pmf.shape}); skipping.")
            continue

        pmfs.append(pmf[:,1])
        x = pmf[:,0]

        if plot_indiv:
            plt.plot(x, pmf[:,1], label=f"Run {n_run}")

    # After loop: show individual plot if requested
    if plot_indiv:
        plt.xlabel(free_energy_step)
        plt.ylabel('PMF (kcal/mol)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if len(pmfs) == 0:
        raise RuntimeError("No PMF data was loaded for any run. Check paths and run identifiers.")

    pmfs_arr = np.vstack(pmfs)            # shape (n_loaded_runs, n_points)
    n_loaded = pmfs_arr.shape[0]

    av = np.mean(pmfs_arr, axis=0)
    # standard error of the mean; use population std (ddof=0) unless you want sample std (ddof=1)
    err = np.std(pmfs_arr, axis=0, ddof=0) / np.sqrt(n_loaded)

    if plot_av:
        plt.figure(figsize=(6,4), dpi=200)
        plt.title(f'{free_energy_step} Average PMF')
        if free_energy_step == 'Boresch':
            x_plot = x
        else:
            x_plot = 10 * x  # convert to Å if needed (original code used 10*x)
        plt.plot(x_plot, av, color='k')
        plt.fill_between(x_plot, av - err, av + err, color='grey', alpha=0.4)
        plt.xlabel(free_energy_step)
        plt.ylabel('PMF (kcal/mol)')
        plt.tight_layout()
        plt.show()

    return x, av, err

def obtain_integrands(forceConstant, free_energy_step, restraint_type, dof=None, equilibration=0, sampling_time=None, run_number=1):
    """
    Obtain the numerator and denominator integrands used to calculate the RMSD free energy contribution
    Parameters
    ----------
    free_energy_step  : str
        Section of the thermodynamic cycle where US simulation is performed
    restraint_type : str
        Type of RMSD restraint applied to the system
    dof : str
        Degree of freedom for which to generate metafile
    equilibration : str or int, Optional, default=0
        Type of data truncation applied ('RED' or int)          
    run_number : int, default=1
        Specify the replicate

    Returns
    -------
    x : array
        CV values
    numerator : array
        exp^(-beta*pmf)
    denominator : array
        exp^-[beta*(pmf + 0.5*forceConstant*x^2)]
    """    
    
    dirpath = obtain_dirpath(free_energy_step, restraint_type, dof, equilibration, sampling_time, run_number)

    pmf = np.loadtxt(f'{dirpath}/pmf.txt')

    mask = np.isfinite(pmf[:,1]) # Boolean mask to remove inf values 
    x, y = pmf[:,0][mask], pmf[:,1][mask]   

    numerator = np.exp(-beta*y)
    denominator = np.exp(-beta*(y+0.5*forceConstant*np.square(x)))

    return x, numerator, denominator

def obtain_av_plateau(x, av, err, plateau_bounds=[2.5, 3.0]):
    """
    Calculate the average plateau value and the average standard error of the plateau region
    """
    if len(plateau_bounds)!=2:
        raise ValueError('Select a start and end value for the plateau region')
    
    start_idx = min(range(len(x)), key=lambda i: abs(x[i] - plateau_bounds[0]))
    end_idx = min(range(len(x)), key=lambda i: abs(x[i] - plateau_bounds[1]))

    return np.average(av[start_idx:end_idx]), np.average(err[start_idx:end_idx])

def RMSDContribution(x, pmf, k_rmsd, unbound=False):
    """
    Calculate the contribution of RMSD restraints
    Make sure units of k_rmsd are consistent
    """
    pmf = np.array([x,pmf])

    # mask = np.isfinite(pmf[:,1]) # Boolean mask to remove inf values 
    # pmf = pmf[:,0][mask], pmf[:,1][mask]

    width = pmf[0][1] - pmf[0][0]

    restraintCenter = 0

    # integration
    numerator = 0
    denominator = 0
    for x, y in zip(pmf[0], pmf[1]):
        numerator += math.exp(-beta * y)
        denominator += math.exp((-beta) * (y + 0.5 * k_rmsd * ((x - restraintCenter)**2)))
    
    contribution = math.log(numerator / denominator) / beta

    # print(f"Numerator is {numerator}\n")
    # print(f"Denominator is {denominator}\n")
    
    if unbound:
        return contribution
    else:
        return -contribution

def BoreschContribution(x, pmf, theta_0, k_restraint):
    """Calculate the free energy contribution from applying a Boresch restraint"""
    pmf = np.array([x,pmf])
    
    width = pmf[0][1] - pmf[0][0]

    restraint_center = theta_0

    numerator = 0
    denominator = 0
    for x, y in zip(pmf[0], pmf[1]):
        numerator += width*math.exp(-beta*y)
        denominator += width*math.exp((-beta)*(y+0.5*k_restraint*((x-restraint_center)**2)))
    
    contribution = math.log(numerator/denominator)/beta

    return -contribution

def standard_state_correction(r_star, theta_a_min, theta_b_min, k_boresch):
    """
    Calculate the free energy contribution for release of the separated proteins to the standard state 
    """
    
    corr = (r_star**2)*math.sin(theta_a_min)*math.sin(theta_b_min)*(2*np.pi/beta)**2.5/(8*(np.pi**2)*(4*np.pi*radius_sphere**2)*(k_boresch)**2.5)

    return -1/(beta)*math.log(corr)

def SepContribution(x, pmf, r_star):
    """
    Integrate the separation PMF to obtain the free energy contribution of separating the restrained proteins
    """

    pmf = np.array([x,pmf])
    
    w_r_star = pmf[1][0]
    for x, y in zip(pmf[0], pmf[1]):
        if x >= r_star:
            w_r_star = y
            break
        
    width = pmf[0][1] - pmf[0][0]
    I = 0
    for x, y in zip(pmf[0], pmf[1]):
        I += width*math.exp(-beta*(y-w_r_star))
        if x >= r_star:
            break

    return -1/(beta)*math.log(3*I/radius_sphere)