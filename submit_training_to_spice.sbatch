#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --export=NONE

conda deactivate
conda activate MLFoundationsEnv
echo "Conda environment 'MLFoundationsEnv' activated"

script=$1
model_type=$2
training_frequency=$3
with_rasterized_ozone=$4
ndays=$5
k_folds=$6

now=$(date -u -Isec)
echo -e '\n'
echo '****************************************************************'
echo "$now: ${script}"
echo '****************************************************************'

# print the submitted script and its arguments for logging purposes
echo -e '\n'
echo "Running script: ${script} "
echo "Model Type: ${model_type}"
echo "--training_frequency=${training_frequency}"
echo "--with_rasterized_ozone=${with_rasterized_ozone}"
echo "--ndays=${ndays}"
echo "--k_folds=${k_folds}"

# run the training script with the specified arguments and measure the execution time
if [ "${with_rasterized_ozone}" == "True" ]; then
    time python "${script}" \
        --model_type "${model_type}" \
        --training_frequency "${training_frequency}" \
        --ndays "${ndays}" \
        --k_folds "${k_folds}" \
        --with_rasterized_ozone
else
    time python "${script}" \
        --model_type "${model_type}" \
        --training_frequency "${training_frequency}" \
        --ndays "${ndays}" \
        --k_folds "${k_folds}"
fi

echo -e '\n'
echo "Finished running"
