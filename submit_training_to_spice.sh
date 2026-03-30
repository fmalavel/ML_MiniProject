# Submit training scripts to SPICE with combinations of:
# - training frequency (daily/hourly)
# - rasterized ozone option (True/False)
#
# Memory is set automatically by frequency:
# - hourly: 150G
# - daily: 20G
#
# Job names are derived from each Python script basename.
#
# Usage:
#   bash submit_training_to_spice.sh

# Link to documentation for SPICE: https://wwwspice/docs/compute/

# create log and trained_models directories if they don't exist
mkdir -p logs
mkdir -p Trained_models
# echo "Directories 'logs' and 'trained_models' created if they did not already exist"
echo "Directories 'logs' and 'trained_models' created if they did not already exist"

submit() {
    # submit a training job to SPICE with the specified script, training frequency, and rasterized ozone option
    local script=$1
    local model_type=$2
    local training_frequency=$3
    local with_rasterized_ozone=$4
    local ndays=$5
    local k_folds=$6

    # convert with_rasterized_ozone to boolean for the training script
    local with_rasterized_ozone_bool
    if [ "${with_rasterized_ozone}" == "with_rasterized_ozone" ]; then
        with_rasterized_ozone_bool=True
    elif [ "${with_rasterized_ozone}" == "met_only" ]; then
        with_rasterized_ozone_bool=False
    fi

    # determine memory & partition based on training frequency
    local MEM
    local partition
    # if hourly mem 150G, if daily mem 20G
        if [ "${training_frequency}" == "hourly" ]; then
            MEM="200G"
            partition="cpu-long"
        else
            MEM="30G"
            partition="cpu"

        fi

    # derive job name from script basename and training parameters 
    local jobname
    jobname=${model_type}"_${training_frequency}_${with_rasterized_ozone}"
    local jobout="logs/${jobname}-%j.out"
    local joberr="logs/${jobname}-%j.err"

    # submit the job to SPICE with the specified parameters and log the submission details
    sbatch \
        --job-name="${jobname}" \
        --partition="${partition}" \
        --mem="${MEM}" \
        --output="${jobout}" \
        --error="${joberr}" \
        submit_training_to_spice.sbatch \
        "${script}" \
        "${model_type}" \
        "${training_frequency}" \
        "${with_rasterized_ozone_bool}" \
        "${ndays}" \
        "${k_folds}"
}

model_types=(
    "MLP"
    "2DCNN"
    "3DCNN"
    "CNN+LSTM"
    "convLSTM"
    "UNet"
)

training_frequencies=(
    "daily"
    # "hourly"
)

with_rasterized_ozone_options=(
    "with_rasterized_ozone"
    "met_only"
)

# set the number of days for training based on the training frequency to manage runtime and memory requirements
ndays_daily=93 # number of days in the dataset for daily frequency
ndays_hourly=30 # number of days for hourly frequency to manage runtime and memory requirements

# number of k-folds for cross-validation (must be >= 2)
k_folds=5

# use a generic training script that can handle different model types, training frequencies, and rasterized ozone options based on command-line arguments
script="Training_generic.py"

for model_type in "${model_types[@]}"; do
    # loop over training frequencies and rasterized ozone options to submit jobs for each combination
    for training_frequency in "${training_frequencies[@]}"; do
        if [ "${training_frequency}" == "hourly" ]; then
            ndays=${ndays_hourly} 
        else
            ndays=${ndays_daily}
        fi
        # loop over rasterized ozone options to submit jobs for each combination of training frequency and rasterized ozone option
        for with_rasterized_ozone in "${with_rasterized_ozone_options[@]}"; do
            submit "${script}" "${model_type}" "${training_frequency}" "${with_rasterized_ozone}" "${ndays}" "${k_folds}"
        done
    done
done
