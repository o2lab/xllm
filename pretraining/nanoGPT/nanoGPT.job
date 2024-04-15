#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=nanoGPT       #Set the job name to "JobExample3"
#SBATCH --time=6:00:00            #Set the wall clock limit to 1 Day and 12hr
#SBATCH --nodes=2                    #Request 1 node
#SBATCH --cpus-per-task=32          #Request 2 tasks/cores per node
#SBATCH --mem=65536M                  #Request 4096MB (4GB) per node 
#SBATCH --output=nanoGPTLOG.%j      #Send stdout/err to "Example3Out.[jobID]"
#SBATCH --gres=gpu:a100:2                 #Request 1 GPU per node cam be 1 or 2


##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=siweicui@tamu.edu    #Send all emails to email_address

#First Executable Line

cd $SCRATCH
module add GCC/10.3.0  OpenMPI/4.1.1
# module add PyTorch/1.10.0
# module add Anaconda3/2022.05
# module add CMake/3.20.1
# module add CUDA/12.2

module load WebProxy
/bin/bash
source activate llm
cd /scratch/user/siweicui/
python /scratch/user/siweicui/override_script.py
cd /scratch/user/siweicui/nanoGPT/
bash run_master.sh

