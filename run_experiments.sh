#!/bin/bash

sbatch exe/vera/vary-expert.sh phi4_10ex amazon
sbatch exe/vera/vary-expert.sh phi4_10ex misinfo
sbatch exe/vera/vary-expert.sh phi4_10ex biobias
sbatch exe/vera/vary-expert.sh phi4_10ex germeval

sbatch exe/vera/vary-total.sh phi4_10ex amazon
sbatch exe/vera/vary-total.sh phi4_10ex misinfo
sbatch exe/vera/vary-total.sh phi4_10ex biobias
sbatch exe/vera/vary-total.sh phi4_10ex germeval
