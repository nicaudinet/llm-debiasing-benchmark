library(rslurm)

test_func <- function(par_mu, par_sd) {
    samp <-rnorm(10^6, par_mu, par_sd)
    c(s_mu = mean(samp), s_sd = sd(samp))
}

pars <- data.frame(
    par_mu = 1:10,
    par_sd = seq(0.1, 1, length.out = 10)
)

head(pars, 3)

    jobname = "test_apply",
    nodes = 2,
    cpus_per_node = 2,


slurm_options <- list(
    account = "C3SE2024-1-17",
    partition = "vera",
    nodes = 2,
    ntasks_per_node = 1,
    cpus_per_task = 32,
    time = "3-00:00:00",
    mail_user = "nicolas.audinet@chalmers.se",
    mail_type = "all",
    output = "logs/output/output_%j.log",
    error = "logs/error/error_%j.log",
)


sjob <- slurm_apply(
    test_func,
    pars,
    jobname = "dsl-use",
    slurm_options = slurm_options,
    submit = TRUE
)

res <- get_slurm_out(
    sjob,
    outtype = 'table',
    wait = TRUE
)

print(res)

cleanup_files(sjob)
