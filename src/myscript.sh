#!/bin/bash
#PBS -N train_asr_model
#PBS -l select=1:ncpus=4:mem=96GB:ngpus=3:Qlist=ee
#PBS -l walltime=24:00:00
#PBS -m ae
#PBS -e output.err
#PBS -o output.out
#PBS -M 22614524@sun.ac.za

# Make sure I'm the only one that can read my output
umask 0077

# Create a temporary directory with the job ID as name in /scratch-small-local
SPACED="${PBS_JOBID//./-}" 
TMP=/scratch-small-local/${SPACED} # E.g. 249926.hpc1.hpc
mkdir -p ${TMP}
echo "Temporary work dir: ${TMP}"
cd ${TMP}

# Copy the input files to ${TMP}
echo "Copying from ${PBS_O_WORKDIR}/ to ${TMP}/"
/usr/bin/rsync -vax "${PBS_O_WORKDIR}/" ${TMP}/

# Load python, create env, install dependencies
module load python/3.8.5
python3 -m pip install --upgrade pip
python3 -m venv myenv	
source myenv/bin/activate
pip install -r requirements.txt

# Install Git-LFS
# curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

# Run training script
mkdir wav2vec2-xls-r-300m-asr_af
python3 train.py

# Job done, deactivate environment and copy everything back
deactivate
echo "Copying from ${TMP}/ to ${PBS_O_WORKDIR}/"
/usr/bin/rsync -vax ${TMP}/ "${PBS_O_WORKDIR}/"

# If the copy back succeeded, delete my temporary files
cd ..
[ $? -eq 0 ] && /bin/rm -rf ${TMP}
