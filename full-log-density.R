library(argparse)
library(bridgestan)
library(cmdstanr)
library(posterior)
library(data.table)

parser <- ArgumentParser(
  prog = "full-log-density",
  description = "Calculate the full log-density at points sampled using an approximation (for subsequent PSIS)"
)

parser$add_argument("stan_file", help = "The Stan source (.stan) or BridgeStan compiled (.so) model")
parser$add_argument("data", help = "The JSON file for the full data")
parser$add_argument("fit_approx", help = "The CmdStan CSV fit to the approximate model")
parser$add_argument("output", help = "The destination for the PSIS samples")
parser$add_argument("--seed", default = 1234, type = "integer", help = "A seed for BridgeStan (should have no effect)")
parser$add_argument("--range", nargs = 2, type = "integer", help = "Range (1-indexed, inclusive) of samples to use (excluding warmup)")

args <- parser$parse_args()

cat("Creating BridgeStan model...\n")
# Compiling from .stan: BridgeStan builds without threads by default, which ran
# about 5 times slower here (reduce_sum then runs serially). A prebuilt .so is
# used as it is
model <- StanModel$new(args$stan_file, args$data, args$seed, make_args = c("STAN_THREADS=true"))
cat(paste0("  The model has ", model$param_num(), " parameters\n"))

cat("Loading CmdStan approximate fit...\n")
fit <- as_cmdstan_fit(args$fit_approx)
cancer <- ""
if ("m_NSCLC" %in% names(fit$metadata()$stan_variable_sizes)) {
  cancer <- "NSCLC"
  n_eta <- fit$metadata()$stan_variable_sizes[["eta"]]
  cat(paste0("  Fit corresponds to an NSCLC model with ", n_eta, " random effect parameters\n"))
} else {
  cancer <- "SCLC"
  cat("  Fit corresponds to an SCLC model\n")
}

# The parameters (excluding transformed parameters and generated quantities)
# in BridgeStan's order, with BridgeStan's names ("logit_sens.1.1") converted to
# CmdStan's ("logit_sens[1,1]"). Stan variable names cannot contain ".".
param_names <- model$param_names()
param_names <- ifelse(
  grepl(".", param_names, fixed = TRUE),
  paste0(gsub(".", ",", sub(".", "[", param_names, fixed = TRUE), fixed = TRUE), "]"),
  param_names
)
missing <- setdiff(param_names, fit$metadata()$variables)
if (length(missing) > 0) {
  stop(paste0(
    "The fit has no column for ", length(missing), " of the model's ", length(param_names),
    " parameters (first: ", missing[1], "). For NSCLC, check that the full data give every ",
    "non-cancer patient wt != 1 (e.g. 1 + 1e-9), so that only the cancer patients have an eta"
  ))
}
if (cancer == "NSCLC" && model$param_num() != 40 + n_eta) {
  stop(paste0("The model has ", model$param_num(), " parameters but the fit has ", 40 + n_eta))
}

iter_sampling <- fit$metadata()$iter_sampling
start <- if (!is.null(args$range)) args$range[1] else 1
end <- if (!is.null(args$range)) args$range[2] else iter_sampling
if (start < 1 || end > iter_sampling || start > end) {
  stop(paste0("--range must satisfy 1 <= start <= end <= ", iter_sampling))
}
cat(paste0("  Using draws ", start, " to ", end, " of ", iter_sampling, " (warmup excluded)\n"))

# Every column of the CmdStan CSV, in its order: lp__, the sampler
# diagnostics, then the parameters, transformed parameters and generated
# quantities
columns <- c(
  "lp__", fit$metadata()$sampler_diagnostics,
  setdiff(fit$metadata()$variables, "lp__")
)
out <- bind_draws(
  fit$sampler_diagnostics(format = "draws_df"),
  fit$draws(format = "draws_df"),
  along = "variable"
) |>
  subset_draws(iteration = start:end) |>
  as.data.frame()
out <- as.data.table(out)[, ..columns]
lp_approx <- out$lp__
draws <- as.matrix(out[, ..param_names])

# Write the columns under the CSV's own names ("logit_sens.1.1"; cmdstanr
# renames them "logit_sens[1,1]"), which are in the same order
con <- file(args$fit_approx, "r")
repeat {
  header <- readLines(con, n = 1)
  if (!startsWith(header, "#")) break
}
close(con)
header <- strsplit(header, ",", fixed = TRUE)[[1]]
if (length(header) != ncol(out)) {
  stop("The CSV header does not match the columns read by cmdstanr")
}
setnames(out, header)

cat("Checking parameter transforms on the first draw...\n")
theta_unc <- model$param_unconstrain(draws[1, ])
round_trip <- model$param_constrain(theta_unc)
rel_err <- max(abs(round_trip - draws[1, ]) / pmax(abs(draws[1, ]), 1e-300))
cat(paste0("  Largest relative error after unconstraining and constraining again: ",
           signif(rel_err, 3), "\n"))
if (rel_err > 1e-6) {
  stop("Unconstraining and constraining again does not reproduce the draw: check the parameter order")
}

# Rows are appended as they are computed, so a long run that stops early keeps
# its finished draws (rerun with --range for the rest)
cat("Calculating the full log-density...\n")
fwrite(out[0, ][, lp_full := numeric(0)], args$output)
n <- nrow(draws)
t0 <- Sys.time()
for (i in seq_len(n)) {
  lp_full <- tryCatch(
    model$log_density(model$param_unconstrain(draws[i, ]), propto = FALSE),
    error = function(e) {
      warning(paste0("Draw ", start + i - 1, ": ", conditionMessage(e), " (lp_full set to NA)"))
      NA_real_
    }
  )
  fwrite(out[i, ][, lp_full := lp_full], args$output, append = TRUE)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  draw %d (%d of %d): lp_approx %.6f, lp_full %.6f, log ratio %.6f; %.1f s per draw, about %.1f h left\n",
              start + i - 1, i, n, lp_approx[i], lp_full, lp_full - lp_approx[i],
              elapsed / i, elapsed / i * (n - i) / 3600))
}
cat(paste0("Wrote ", n, " draws to ", args$output, "\n"))
