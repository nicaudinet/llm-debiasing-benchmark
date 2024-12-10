
# Compute the data for the DSL use experiment
# Saves the data in a file on disk (filename argument required)

#############
## Imports ##
#############

library(MASS)
library(dsl)
library(doParallel)
library(foreach)
library(fields)

###############
## Arguments ##
###############

# Command line arguments

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("No filename argument found. Please provide one in first position") 
}
data_file <- args[1]

# Cannot change

CL <- 0.95 # confidence level
num_coeffs <- 4 + 1 # independent variables + intercept

# Can change

pred_accuracy <- 0.9 # "LLM" prediction accuracy
NN <- 10 # side length of the grid (2 <= NN)
ncores <- as.numeric(Sys.getenv("SLURM_CPUS_ON_NODE"))
# ncores <- 4

###############
## Functions ##
###############

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
    num_samples <- ceiling(10^seq(from = 2, to = 4, length.out = NN))
    dim <- c(NN, NN, num_coeffs)
    size <- prod(dim)
    local_coeffs_true <- array(rep(0, size), dim = dim)
    local_coeffs_exp <- array(rep(0, size), dim = dim)
    local_stderr_exp <- array(rep(0, size), dim = dim)
    local_coeffs_dsl <- array(rep(0, size), dim = dim)
    local_stderr_dsl <- array(rep(0, size), dim = dim)

    for (i in 1:NN) {

        for (j in 1:i) {

            # Generate data
            n <- num_samples[i]
            data <- generate(n)

            # Compute true coefficients
            out_true <- fit(data, 1:n)
            local_coeffs_true[i, j, ] <- out_true$coefficient

            # Select samples for expert annotation
            selected <- sample(1:n, size = num_samples[j], replace = FALSE)

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

##########
## Main ##
##########

# Setup parallel backend to use multiple processors
print(paste("Run simulations in parallel on ", ncores, " cores"))
registerDoParallel(cl <- makeCluster(ncores))

# Run the simulation in parallel
results <- foreach(
    r = 1:R,
    .packages = c("dsl", "MASS"),
    .inorder = FALSE, # don't need the results in order
    .errorhandling="remove" # don't include NA results
) %dopar% {
    tryCatch(
        simulate(),
        warning = function(w) {
            message("A warning occurred")
            print(w)
            return(NA)
        },
        error = function(e) {
            message("An error occurred")
            print(e)
            return(NA)
        }
    )
}

# Initialise final arrays
result_size <- length(results)
dim <- c(result_size, NN, NN, num_coeffs)
size <- prod(dim)
coeffs_true <- array(rep(0, size), dim = dim)
coeffs_exp <- array(rep(0, size), dim = dim)
stderr_exp <- array(rep(0, size), dim = dim)
coeffs_dsl <- array(rep(0, size), dim = dim)
stderr_dsl <- array(rep(0, size), dim = dim)

# Aggregate results back into the arrays
for (r in 1:result_size) {
    result <- results[[r]]
    coeffs_true[r, , , ] <- result$coeffs_true
    coeffs_exp[r, , , ] <- result$coeffs_exp
    stderr_exp[r, , , ] <- result$stderr_exp
    coeffs_dsl[r, , , ] <- result$coeffs_dsl
    stderr_dsl[r, , , ] <- result$stderr_dsl
}

# Stop parallel backend
stopCluster(cl)

# Save results in a file
save(
    coeffs_true,
    coeffs_exp,
    stderr_exp,
    coeffs_dsl,
    stderr_dsl,
    file = data_file
)
