#!/bin/bash
# This is a template to show how to train RiskSLIM from a Bash command shell
# You should adapt this to run RiskSLIM on a batch computing environment (e.g. AWS Batch)
#
# To test the script, run the following command from risk-slim directory:
#
# `bash batch/job_template.sh`
#
# To see a detailed list of all arguments that can be passed into risk_slim, use:
#
# `python "batch/train_risk_slim.py --help`
#
#  or
#
# `python2 "batch/train_risk_slim.py --help`
#
# Recommended Directory Structure for Batch Computing:
#
#   risk-slim/
#       └──batch/
#           └──data/           location of CSV files for data (ignored in git)
#           └──logs/           directory where log files are printed out (ignored in git)
#           └──results/        directory where results files are stored (ignored in git)
#       └──doc/
#       └──examples/           
#       └──riskslim/           directory where code is stored (do not change this to be able to pull from GitHub)
#       └──setup.py
#
# Advantaged settings are be configured through a JSON file. See: `batch/settings_template.json` for a template
# The values can be changed directly using a text editor, or programmatically using a tool such as
# `jq` https://stedolan.github.io/jq/

#directories
repo_dir=$(pwd)
data_dir="${repo_dir}/examples/data"    #change to /batch/data/ for your own data
batch_dir="${repo_dir}/batch"
results_dir="${batch_dir}/results"
log_dir="${batch_dir}/logs"

#set job parameters
data_name="breastcancer"
data_file="${data_dir}/${data_name}_data.csv"

cvindices_file="${data_dir}/${data_name}_cvindices.csv"
#weights_file="${data_dir}/${data_name}_weights.csv"
fold=0

max_coef=5
max_size=5
max_offset=-1
w_pos=1.00
c0_value=1e-6

timelimit=60

#results_file and log_file must have a UNIQUE name for each job to avoid overwriting existing files
run_name="${data_name}_fold_${fold}"
run_time=$(date +"%m_%d_%Y_%H_%M_%S")
results_file="${results_dir}/${run_name}_results.p"
log_file="${log_dir}/${run_name}_${run_time}.log"

#comment out the following in case testing / OK with overwriting
#for safety, train_risk_slim.py will not run if results_file exists on disk
rm -f "${results_file}" #c

#create directories that do not exist
mkdir -p "${results_dir}"
mkdir -p "${log_dir}"

#addition settings can be modified by changing a JSON file
#complete list of settings is in: risk-slim/batch/settings_template.json
settings_file="${results_dir}/${run_name}_settings.json"
cp "${batch_dir}/settings_template.json" "${settings_file}"

#run command
python3 "${batch_dir}/train_risk_slim.py"  \
    --data "${data_file}" \
    --results "${results_file}" \
    --cvindices "${cvindices_file}" \
    --fold "${fold}" \
    --timelimit "${timelimit}" \
    --settings "${settings_file}" \
    --w_pos "${w_pos}" \
    --c0_value "${c0_value}" \
    --max_size "${max_size}" \
    --max_coef "${max_coef}" \
    --max_offset "${max_offset}" \
    --log "${log_file}"

exit
W