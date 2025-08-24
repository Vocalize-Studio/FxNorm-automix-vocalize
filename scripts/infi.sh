#bash
#!/usr/bin/bash

parent_path=$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd -P
)
cd "$parent_path"

export CUDA_VISIBLE_DEVICES=0

OUTPUT_FOLDER="../mixes"
CONFIG_FOLDER="../configs/ISMIR"

PATH_IR
