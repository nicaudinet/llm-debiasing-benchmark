#############
## Imports ##
#############

library(fields)

###############
## Arguments ##
###############

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Missing data_file and/or plot_file argument(s)") 
}
data_dir <- args[1]
plot_file <- args[2]

#################
## Gather data ##
#################

files <- list.files(data_dir, pattern = "data_.+", full.names = TRUE)

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

########################
## Compute statistics ##
########################

epsilon <- max

calc_bias <- function(Y, Y_hat) {
    error_std <- (Y - Y_hat) / Y
    return(apply(error_std, MARGIN = c(2,3), FUN = mean))
}

# Compute the standard bias
bias_exp <- calc_bias(coeffs_true, coeffs_exp)
bias_dsl <- calc_bias(coeffs_true, coeffs_dsl)
error <- bias_exp / bias_dsl
NN <- dim(coeffs_true)[2]
error <- matrix(error[nrow(error):1, ncol(error):1], NN, NN)
error[col(error) < row(error)] <- NA

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

error[is.nan(error)] <- NA

print(coeffs_true)
print(coeffs_exp)
print(bias_exp)
print(bias_dsl)
print(error)

##################
## Plot results ##
##################

# Open PDF device
pdf(plot_file, width = 14, height = 10)

# Plot image and legend
image.plot(
    new = FALSE,
    z = error,
    xlab = "Number of expert samples",
    ylab = "Number of total samples",
    col = colorRampPalette(c("white", "red"))(100),
    useRaster = TRUE,
    axes = FALSE,
    zlim = range(error, na.rm = TRUE)
)

# Set NAs to grey
if (any(is.na(error))) {
    mmat <- ifelse(is.na(error), 1, NA)
    image(
        mmat,
        axes = FALSE,
        col = "grey",
        useRaster = TRUE,
        add = TRUE
    )
}

title("Error")

# Plot image axes
at = c(0,0.5,1)
labels = c(expression(10^2), expression(10^3), expression(10^4))
axis(1, at = at, labels = labels)
axis(2, at = at, labels = labels)

# Surround image with box
box()

# Save plot to image
dev.off()
