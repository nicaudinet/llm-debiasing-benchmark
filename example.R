args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Missing data_file and/or plot_file argument(s)") 
}
data_dir <- args[1]

#################
## Gather data ##
#################

files <- list.files(data_dir, pattern = "\\.RData$", full.names = TRUE)

loadRData <- function(filename) {
	load(filename)
	return(list(
		coeffs_true = coeffs_true,
		coeffs_exp = coeffs_exp,
		coeffs_dsl = coeffs_dsl,
		stderr_exp = stderr_exp,
		stderr_dsl = stderr_dsl
	))
}

data <- loadRData(files[1])
dims <- dim(data[["coeffs_true"]])
num_reps <- dims[1]
num_files <- length(files)
dims <- c(num_files * num_reps, dims[2], dims[3], dims[4])
size <- prod(dims)

coeffs_true <- array(rep(0, size), dim = dims)
coeffs_exp <- array(rep(0, size), dim = dims)
stderr_exp <- array(rep(0, size), dim = dims)
coeffs_dsl <- array(rep(0, size), dim = dims)
stderr_dsl <- array(rep(0, size), dim = dims)

for (i in seq_along(files)) {
	data <- loadRData(files[i])
	start <- num_reps * (i - 1)
	end <- num_reps * i
	print(paste(files[i], " (", start, ":", end, ")", sep = ""))
	coeffs_true[start:end, , , ] <- data[["coeffs_true"]]
	coeffs_exp[start:end, , , ] <- data[["coeffs_exp"]]
	coeffs_dsl[start:end, , , ] <- data[["coeffs_dsl"]]
	stderr_exp[start:end, , , ] <- data[["stderr_exp"]]
	stderr_dsl[start:end, , , ] <- data[["stderr_dsl"]]
}
