#!/bin/bash

set -eo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/venv_alvis/bin/activate
python -m pip install --no-cache-dir -r requirements_cluster.txt
