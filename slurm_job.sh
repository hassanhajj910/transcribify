#!/bin/sh
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -G gtx1080:2
#SBATCH --mem=50G
#SBATCH -o outfile.%J
#SBATCH --mail-user=hhajj@mpiwg-berlin.mpg.de
#SBATCH -t 12:00:00
module load anaconda3
source $ANACONDA3_ROOT/etc/profile.d/conda.sh
conda activate ml_audio_small
srun python transcribify.py -i 'data/imprs_workshop' -o 'results/imprs_workshop'