library(MASS)
library(dsl)
library(doParallel)
library(foreach)
library(fields)

ncores <- as.numeric(Sys.getenv("SLURM_CPUS_ON_NODE"))
print(ncores)

cl <- makeCluster(ncores)
registerDoParallel(cl)

result <- foreach(i = 1:10, .combine = c) %dopar% {
    Sys.sleep(0.1)
    sqrt(i)
}

write.csv(result, file = "dummy.csv")

stopCluster(cl)
