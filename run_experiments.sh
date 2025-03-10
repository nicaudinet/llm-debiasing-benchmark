#!/bin/bash

sbatch exe/vera/vary-expert-simulation.sh
sbatch exe/vera/vary-expert-amazon.sh
sbatch exe/vera/vary-expert-misinfo.sh
sbatch exe/vera/vary-expert-biobias.sh
sbatch exe/vera/vary-expert-germeval.sh

sbatch exe/vera/vary-total-simulation.sh
sbatch exe/vera/vary-total-amazon.sh
sbatch exe/vera/vary-total-misinfo.sh
sbatch exe/vera/vary-total-biobias.sh
sbatch exe/vera/vary-total-germeval.sh
