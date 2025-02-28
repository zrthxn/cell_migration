#!/bin/bash
# ^Batch script starts with shebang line

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=52
#SBATCH --time=23:00:00               	# specify Slurm options
#SBATCH --account=p_planarian         	#
#SBATCH --job-name=morpheus-mldens    	# All #SBATCH lines have to follow uninterrupted
#SBATCH --output=aus.txt    		# after the shebang line
#SBATCH --error=err.txt     		# Comments start with # and do not count as interruptions


module load Miniconda3/23.10.0-1

srun /bin/bash ./runme.sh
 