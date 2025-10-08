from pymol import cmd
import sys
import os
import numpy as np
import shutil
import subprocess

r0_values = [0.9, 0.95, 1.0, 1.05, 1.1 , 1.15, 1.2 , 1.25, 1.3 , 1.35, 1.4 , 1.45, 1.5 , 1.55, 1.6 , 1.65, 1.7 , 1.75, 1.8 , 1.85, 1.9 , 1.95, 2.0  , 2.05, 2.1 , 2.2 , 2.3 , 2.4 , 2.5 , 2.6 , 2.7 , 2.8 , 2.9 , 3.0 , 3.1 , 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]

for r_0 in r0_values:

    r_0 = np.round(r_0, 4)
    filepath = f'{r_0}'

    print('\n-----------------------------------------------')
    print(f"Starting input preparation for r0 = {r_0} nm\n")

    """Save lenalidomide input structure"""
    # Load the PDB file
    cmd.reinitialize()
    cmd.load(f"{filepath}/{r_0}.pdb", "complex")

    # Remove waters, hydrogens, and chlorine ions
    cmd.remove('not resn LVY')
    cmd.save(f"{filepath}/len.pdb")

    # Convert to sdf format
    convert_to_sdf = 'obabel len.pdb -O len.sdf'
    subprocess.run(convert_to_sdf, check=True, shell=True, cwd=filepath)

    # Convert to mol2 (AM1-BCC charges)
    convert_to_mol2 = 'antechamber -i len.sdf -fi sdf -o len.mol2 -fo mol2 -at gaff2 -c bcc -rn LVY -s 2'
    subprocess.run(convert_to_mol2, check=True, shell=True, cwd=filepath)

    """Generate frcmod file"""

    generate_frcmod = 'parmchk2 -i len.mol2 -f mol2 -o len.frcmod -s 2'
    subprocess.run(generate_frcmod, check=True, shell=True, cwd=filepath)

    """Remove all hydrogen atoms, water molecules and chlorine ions from the system"""

    # Load the PDB file
    cmd.reinitialize()
    cmd.load(f"{filepath}/{r_0}.pdb", "complex")

    # Remove waters, hydrogens, and chlorine ions
    cmd.remove("resn HOH")
    cmd.remove("hydro")
    cmd.remove("resn CL")
    cmd.remove("resn LVY") # Remove lenalidomide

    # Save the modified structure temporarily
    cmd.save(f"{filepath}/temp.pdb")

    """Add TER between residues 280 and 281"""

    with open(f"{filepath}/temp.pdb", "r") as infile, open(f"{filepath}/protein_noH.pdb", "w") as outfile:
        last_residue = None
        for line in infile:
            # Skip lines starting with 'CONECT' or 'CONNECT'
            if line.startswith("CONECT") or line.startswith("CONNECT"):
                continue

            # Replace HETATOM with ATOM
            if "HETATM" in line and " ZN " in line:
                line = line.replace("HETATM", "ATOM  ", 1)

            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_num = int(line[22:26].strip())
                # Insert a TER line where needed
                if last_residue in [137, 313] and res_num in [138, 314]:
                    outfile.write("TER\n")
                last_residue = res_num
            # Write the line to the output file
            outfile.write(line)       

    # Clean up
    os.remove(f"{filepath}/temp.pdb")

    """Generate tleap.in file"""

    # Copy over frcmod and prep files
    shutil.copy('ZAFF.frcmod', f'{filepath}/ZAFF.frcmod')
    shutil.copy('ZAFF.prep', f'{filepath}/ZAFF.prep')

    # Write tleap lines
    input_filename = f"{r_0}_modified.pdb"

    line0 = 'source leaprc.protein.ff19SB'
    line1 = 'source leaprc.water.tip3p'
    line2 = 'addAtomTypes { { "ZN" "Zn" "sp3" } { "S3" "S" "sp3" } } #Add atom types for the ZAFF metal center with Center ID 1'
    line3 = 'loadamberparams frcmod.ions1lm_126_tip3p'
    line4 = 'loadamberprep ZAFF.prep #Load ZAFF prep file'
    line5 = 'loadamberparams ZAFF.frcmod #Load ZAFF frcmod file\n'

    line6 = 'source leaprc.gaff2'
    line7 = 'loadAmberParams "len.frcmod"'

    line8 = f'protein = loadpdb protein_noH.pdb'

    line9 = 'bond protein.313.ZN protein.210.SG'
    line10 = 'bond protein.313.ZN protein.213.SG'
    line11 = 'bond protein.313.ZN protein.278.SG'
    line12 = 'bond protein.313.ZN protein.281.SG '

    line13 = 'ligand = loadMol2 "len.mol2"'

    line14 = 'complex = combine {protein ligand}'
    line15 = 'check complex'

    line16 = 'solvateOct complex TIP3PBOX 15.0\n'

    line17 = 'addions complex CL 0\n'

    line18 = 'savepdb complex system_solvated.pdb #Save the pdb file'
    line19 = f'saveamberparm complex system.prmtop system.inpcrd #Save the topology and coordinate files'
    line20 = 'quit #Quit tleap'

    lines = [
        line0,
        line1,
        line2,
        line3,
        line4,
        line5,
        line6,
        line7,
        line8,
        line9,
        line10,
        line11,
        line12,
        line13,
        line14,
        line15,
        line16,
        line17,
        line18, 
        line19,
        line20
    ]

    with open(f'{filepath}/tleap.in', 'w') as file:
        for line in lines:
            file.write(line + '\n')

    # Specify the command and directory
    command = 'tleap -f tleap.in'

    try:
        # Run the command within the specified directory
        subprocess.run(command, check=True, shell=True, cwd=filepath)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")