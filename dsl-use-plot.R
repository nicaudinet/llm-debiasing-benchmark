library(fields)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Missing data_file and/or plot_file argument(s)") 
}
data_file <- args[1]
plot_file <- args[2]

# Load the data
load(data_file)

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
