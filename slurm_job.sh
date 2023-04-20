#!/bin/sh
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -G 1
#SBATCH --mem=50G
#SBATCH -o outfile.%J
#SBATCH --mail-user=hhajj@mpiwg-berlin.mpg.de
#SBATCH -t 12:00:00
module load anaconda3
source $ANACONDA3_ROOT/etc/profile.d/conda.sh
conda activate ml_audio_small
srun python transcribify.py -i 'podcast_ep1.wav' -o 'podcast_ep1.txt'