#!/bin/bash

# Set the TORCH environment variable to the PyTorch version
TORCH=$(python -c "import torch; print(torch.__version__)")

# Export the TORCH environment variable
export TORCH

# Install the required Python packages
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch_geometric==2.0.4
pip install -q class_resolver
pip install -r requirements.txt
pip install git+https://github.com/bowang-lab/MedSAM.git

model_id="1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_&confirm=t"
gdown $model_id