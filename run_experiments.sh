#!/bin/bash

sbatch exe/vera/vary-expert.sh phi4 amazon
sbatch exe/vera/vary-expert.sh phi4 misinfo
sbatch exe/vera/vary-expert.sh phi4 biobias
sbatch exe/vera/vary-expert.sh phi4 germeval

sbatch exe/vera/vary-total.sh phi4 amazon
sbatch exe/vera/vary-total.sh phi4 misinfo
sbatch exe/vera/vary-total.sh phi4 biobias
sbatch exe/vera/vary-total.sh phi4 germeval
