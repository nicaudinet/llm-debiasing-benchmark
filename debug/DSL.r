library(jsonlite)
library(dsl)

data <- fromJSON("misinfo.json")
head(data)

cat("Number of samples:", nrow(data), "\n")
cat("Class balance:", sum(data$y) / nrow(data), "\n")

SEED <- 0
run <- function(data) {

    cat("Setting the seed\n")
    set.seed(SEED)

    cat("Training the model with gold labels for all samples\n")
    M_true <- glm(
        y ~ x1 + x2 + x3 + x4,
        data = data,
        family = binomial
    )

    cat("Training the model with DSL\n")
    M_dsl <- dsl(
        model = "logit",
        formula = y ~ x1 + x2 + x3 + x4,
        predicted_var = "y",
        prediction = "y_hat",
        data = data,
        seed = SEED
    )

    cat("Comparing the two models\n")
    true_coeffs = M_true$coefficients
    dsl_coeffs = M_dsl$coefficients
    print(dsl_coeffs)
    print(true_coeffs)
    cat("RMSE:", sqrt(mean((dsl_coeffs - true_coeffs)^2)), "\n")
    return(list("gold_model" = M_true, "dsl_model" = M_dsl))

}

models <- run(data)

summary(models$gold_model)

summary(models$dsl_model)
