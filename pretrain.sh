#!/bin/bash
#DSUB -n 300k_sampled
#DSUB -A root.bingxing2.gpuuser775
#DSUB -q root.default
#DSUB -l wuhanG5500
#DSUB --job_type cosched
#DSUB -R 'cpu=6;gpu=1;mem=48000'
#DSUB -N 1
#DSUB -e %J.out
#DSUB -o %J.out

module load cuda/12.1 
conda init
conda activate scGPT

STATE_FILE="state_${BATCH_JOB_ID}"
/usr/bin/touch ${STATE_FILE}

function gpus_collection(){
    while [[ `cat "${STATE_FILE}" | grep "over" | wc -l` == "0" ]]; do
        /usr/bin/sleep 1
        /usr/bin/nvidia-smi >> "gpu_${BATCH_JOB_ID}.log"
    done
}
gpus_collection &


python /home/bingxing2/gpuuser775/yueran.sun/pretraining/main.py 445 0.5

echo "over" >> "${STATE_FILE}"

