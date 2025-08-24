#bash
#!/usr/bin/bash
# Script to evaluate automix nets

parent_path=$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd -P
)
cd "$parent_path"

export CUDA_VISIBLE_DEVICES=1

OUTPUT_FOLDER="../mixes"          # Folder name where mixes will be created
CONFIGS_FOLDER="../configs/ISMIR" # "/path to folder with configs files"

# The following are the paths to the folders containing the impulse responses.
# The data loader expects each IR to be in an individual folder and named impulse_response.wav
# e.g. /path/to/IR/impulse-response-001/impulse_response.wav
# Stereo or Mono are supported
PATH_IR="/path/to/IR"         # IRs for data augmentation
PATH_PRE_IR="/path/to/PRE_IR" # IRs for data pre-augmentation of dry stems

mkdir -p "${OUTPUT_FOLDER}"

MODELS_FOLDER="../trainings/results"
PATH_FEATURES="../trainings/features" # Path to average features file

NET="ours_S_Lb" # Model name

# Inference for automatic mixing of vocals and instrumental
# IMPORTANT: Please replace the placeholder file names below with your actual file names.
python ../automix/inference.py --vocals ../mixes/vocals/AllOfMe_converted_by_AOMFemale.wav \
  --other ../mixes/instrumental/allofme_instrumental.wav \
  --output ../mixes/mix_vocals_instrumental.wav \
  --training-params ${CONFIGS_FOLDER}/${NET}.py \
  --nets ${MODELS_FOLDER}/${NET}/net_mixture.dump \
  --weights ${MODELS_FOLDER}/${NET}/current_model_for_mixture.params \
  --features ${PATH_FEATURES}/features_MUSDB18.npy
