"""Script for generating a parameter file for a given job"""
import random 
import os
import sys
import shutil

# Check if jobs directory exists
if not os.path.exists('jobs'):
    os.makedirs('jobs')

# Create list of all submitted job IDs
jobfiles = os.listdir('jobs')
jobids = []

for jobfile in jobfiles:
    jobfile.strip()
    jobids.append(int(jobfile.split('.')[0]))

# Create new params.in copy with a different job id
for n in range(10):

    n = random.randint(100000,999999)

    if n not in jobids:
        shutil.copy('params.in', f'jobs/{n}.in')

        # Save the generated jobid n in a .txt file
        with open('jobid.txt', 'w') as f:
            f.write(str(n))

        break

