library(MASS)
library(dsl)
library(doParallel)
library(foreach)
library(fields)

R <- 100 # number of repetitions for coverage
CL <- 0.95 # confidence level
pred_accuracy <- 0.9 # "LLM" prediction accuracy
num_coeffs <- 4 + 1 # 4 independent variables + intercept
NN <- 10 # side length of the grid (2 <= NN)

data_file <- "dsl_use_data.RData"
plot_file <- "dsl_use_plot.RData"

num_samples <- ceiling(10^seq(from = 2, to = 4, length.out = NN))

expit <- function(x) {
    return(exp(x) / (1 + exp(x)))
}

# Generate data following Appendix E in the original paper
generate <- function(N) {

    # Generate covariates
    cov <- matrix(0.3, nrow = 7, ncol = 7)
    diag(cov) <- 1
    X <- mvrnorm(N, mu = rep(0, 7), Sigma = cov)
    X[, 2] <- as.numeric(X[,2] > qnorm(0.8))

    # Generate outcome
    W <- 0.1 / (1 + exp(0.5*X[,3] - 0.5*X[,2])) +
        1.3*X[,4] / (1 + exp(-0.1*X[,2])) +
        1.5*X[,4]*X[,6] +
        0.5*X[,1]*X[,2] +
        1.3*X[,1] +
        X[,2]
    Y <- rbinom(N, size = 1, prob = expit(W))

    P <- rbinom(N, size = 1, prob = pred_accuracy)
    Y_hat <- P * Y + (1 - P) * (1 - Y)

    # Gather data into data frame
    return(data.frame(
        c1 = X[,1],
        c2 = X[,1]^2,
        c3 = X[,2],
        c4 = X[,4],
        Y = Y,
        Y_hat = Y_hat
    ))

}

# Fit coefficients on the true outcome Y
fit <- function(data, selected) {

    # Train the model
    model <- glm(
        Y ~ c1 + c2 + c3 + c4,
        data = data[selected, ],
        family = binomial
    )

    # Return model parameters and errors
    return(data.frame(
        coefficient = coef(model),
        stderror = sqrt(diag(vcov(model)))
    ))
}

# Fit coefficients on the predicted outcome Y_hat correcting for errors with DSL
fit_dsl <- function(data, selected) {

    # Select samples for expert annotation
    data$Y[-selected] <- NA

    # Fit the coefficients with DSL
    out <- dsl(
        model = "logit",
        formula = Y ~ c1 + c2 + c3 + c4,
        predicted_var = "Y",
        prediction = "Y_hat",
        data = data,
        seed = Sys.time()
    )

    return(data.frame(
        coefficient = out$coefficients,
        stderror = out$standard_errors
    ))
}


# Run a single instance of the simulation
simulate <- function() {

    # Initialise arrays
    dim <- c(NN, NN, num_coeffs)
    size <- prod(dim)
    local_coeffs_exp <- array(rep(0, size), dim = dim)
    local_stderr_exp <- array(rep(0, size), dim = dim)
    local_coeffs_dsl <- array(rep(0, size), dim = dim)
    local_stderr_dsl <- array(rep(0, size), dim = dim)

    for (i in 1:NN) {

        # Generate data
        n <- num_samples[i]
        data <- generate(n)

        # Compute true coefficients
        local_coeffs_true <- (fit(data, 1:n))$coefficient

        for (j in 1:i) {

            # Select samples for expert annotation
            selected <- sample(1:n, size = num_samples[j])

            # Compute coefficients from expert annotations only
            out_exp <- fit(data, selected)
            local_coeffs_exp[i, j, ] <- out_exp$coefficient
            local_stderr_exp[i, j, ] <- out_exp$stderror

            # Compute coefficients with DSL
            out_dsl <- fit_dsl(data, selected)
            local_coeffs_dsl[i, j, ] <- out_dsl$coefficient
            local_stderr_dsl[i, j, ] <- out_dsl$stderror

        }
    }

    return(list(
        coeffs_true = local_coeffs_true,
        coeffs_exp = local_coeffs_exp,
        stderr_exp = local_stderr_exp,
        coeffs_dsl = local_coeffs_dsl,
        stderr_dsl = local_stderr_dsl
    ))

}

# Setup parallel backend to use multiple processors
print("Starting to run simulations in parallel")
ncores <- as.numeric(Sys.getenv("SLURM_CPUS_ON_NODE"))
print(ncores)
registerDoParallel(cl <- makeCluster(ncores))

# Run the simulation in parallel
results <- foreach(r = 1:R, .packages = c("dsl", "MASS")) %dopar% { simulate() }

# Initialise final arrays
dim <- c(R, NN, NN, num_coeffs)
size <- prod(dim)
coeffs_true <- array(rep(0, size), dim = dim)
coeffs_exp <- array(rep(0, size), dim = dim)
stderr_exp <- array(rep(0, size), dim = dim)
coeffs_dsl <- array(rep(0, size), dim = dim)
stderr_dsl <- array(rep(0, size), dim = dim)

# Aggregate results back into the arrays
for (r in 1:R) {
    for (i in 1:NN) {
        coeffs_true[r,i, , ] <- results[[r]]$coeffs_true
    }
    coeffs_exp[r, , , ] <- results[[r]]$coeffs_exp
    stderr_exp[r, , , ] <- results[[r]]$stderr_exp
    coeffs_dsl[r, , , ] <- results[[r]]$coeffs_dsl
    stderr_dsl[r, , , ] <- results[[r]]$stderr_dsl
}

# Stop parallel backend
stopCluster(cl)

save(
    coeffs_true,
    coeffs_exp,
    stderr_exp,
    coeffs_dsl,
    stderr_dsl,
    file = data_file
)

calc_bias <- function(Y, Y_hat) {
    error_std <- (Y - Y_hat) / Y
    return(apply(error_std, MARGIN = c(2,3), FUN = mean))
}

bias_exp <- calc_bias(coeffs_true, coeffs_exp)
bias_dsl <- calc_bias(coeffs_true, coeffs_dsl)
error <- bias_exp / bias_dsl
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
