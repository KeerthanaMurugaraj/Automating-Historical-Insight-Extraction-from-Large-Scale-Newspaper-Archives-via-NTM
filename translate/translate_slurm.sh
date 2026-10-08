#!/bin/bash -l

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --time=10:00:00
#SBATCH -p batch
#SBATCH  --ntasks-per-node=1

LOGFILE=translate_$1_$2.out
DATA_DIR=/path/to/dataset
IP_FILE=${DATA_DIR}/file_name.pkl
COLUMNS=content_clean,title_clean  # add the column name here
OP_DIR=$DATA_DIR

cd /path/to/thisproject/python/translate/
conda activate topic_modeling     # activate your conda
srun python translate_content.py -f $1 -l $2 -i $IP_FILE -o $OP_DIR -t $COLUMNS &> ${LOGFILE}
