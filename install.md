# Steps to run on Vera (mostly)

1) Put local lib into .Renviron:

```bash
echo "R_LIBS=/my/local/lib" >> ~/.Renviron
```

2) Pull R docker image with apptainer

```bash
apptainer pull docker://rocker/r-ver:4.4
mv r-ver_4.4.sif container.sif
```

3) Open a shell in the container

```bash
apptainer shell container.sif
```

4) Install the following R packages:

```R
install.packages('doParallel')
install.packages('MASS')
install.packages('fields')
install.packages('devtools')

library(devtools)
install_github('naoki_egami/dsl', dependencies = TRUE)
```

5) Run the job script

```bash
sbatch run.vera.slurm
```
