library(MASS)
library(dsl)
library(doParallel)
library(foreach)
library(fields)

cl <- makeCluster(4)
registerDoParallel(cl)

simulate <- function(i) {
    Sys.sleep(0.1)
    if (i == 8) {
        if (NA) print("hello")
    }
    sqrt(i)
}

warning_handler <- function(w) {
    message("A warning occurred")
    print(w)
    return(NA)
}

error_handler <- function(e) {
    message("An error occurred")
    print(e)
    return(NA)
}

result <- foreach(i = 1:10, .combine = c, .errorhandling="pass") %dopar% {
    tryCatch({ simulate(i) }, warning = warning_handler, error = error_handler)
}

print(result)

write.csv(result, file = "dummy.csv")

stopCluster(cl)
