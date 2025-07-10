# Clearing variables
rm(list = ls())
library(glmnet)
source('functions.R')

n1 <- 10
n0 <- 40
r <- 1
p <- 3
alpha <- 1.0
lam <- 0.2
threshold <- 1e-4
n_rep <- 2

# Start timer
start_time <- Sys.time()

# Initialize accumulators for each metric
rmse_indiv <- rmse_aggreg <- bias <- dens <- magnitude <- matrix(NA, nrow = 6, ncol = n_rep)

for (t in 1:n_rep) {
  # generate the covariates
  X1 <- matrix(runif(n1 * p, 0.1, 0.9), ncol = p)
  X0 <- matrix(sqrt(runif(n0 * p)), ncol = p)
  # calculate the value of beta
  X <- rbind(X1, X0)
  power_sum <- rowSums(X^r)
  beta <- sqrt(var(power_sum))
  # generate y
  Yt1 <- power_sum / beta + rnorm(n1 + n0) # period 1
  Yt2 <- power_sum / beta + rnorm(n1 + n0) # period 2
  Wln <- solve_w(X1, X0, 1, norm = 'L-infinity')
  Wl1 <- solve_w(X1, X0, 1, norm = 'L1')
  Wl2 <- solve_w(X1, X0, 1, norm = 'L2')
  # estimate treatment effects
  tau_ln <- Yt2[1:n1] - t(Wln) %*% Yt2[(n1+1):(n1+n0)]
  tau_l1 <- Yt2[1:n1] - t(Wl1) %*% Yt2[(n1+1):(n1+n0)]
  tau_l2 <- Yt2[1:n1] - t(Wl2) %*% Yt2[(n1+1):(n1+n0)]
  
  # summarize the results
  # RMSE individual
  rmse_indiv_ln <- mean(tau_ln^2)
  rmse_indiv_l1 <- mean(tau_l1^2)
  rmse_indiv_l2 <- mean(tau_l2^2)
  
  # RMSE aggregate
  rmse_aggreg_ln <- mean(tau_ln)^2
  rmse_aggreg_l1 <- mean(tau_l1)^2
  rmse_aggreg_l2 <- mean(tau_l2)^2
  
  # bias
  bias_ln <- abs(mean(tau_ln))
  bias_l1 <- abs(mean(tau_l1))
  bias_l2 <- abs(mean(tau_l2))
  
  # density
  dens_ln <- mean(Wln > threshold)
  dens_l1 <- mean(Wl1 > threshold)
  dens_l2 <- mean(Wl2 > threshold)
  
  # magnitude
  mag_ln <- max(Wln)
  mag_l1 <- max(Wl1)
  mag_l2 <- max(Wl2)
  
  # Ridge regression model
  ridge_model <- glmnet(X0, Yt2[(n1+1):(n1+n0)], alpha = 0) 
  cv_model <- cv.glmnet(X0, Yt2[(n1+1):(n1+n0)], alpha = 0)
  Yt2_corr <- Yt2 - predict(ridge_model, newx = X, s = cv_model$lambda.min)
  
  # Estimate treatment effects
  tau_ln_bc <- Yt2_corr[1:n1] - t(Wln) %*% Yt2_corr[(n1+1):(n1+n0)]
  tau_l1_bc <- Yt2_corr[1:n1] - t(Wl1) %*% Yt2_corr[(n1+1):(n1+n0)]
  tau_l2_bc <- Yt2_corr[1:n1] - t(Wl2) %*% Yt2_corr[(n1+1):(n1+n0)]
  
  # Summarize the results
  # RMSE individual
  rmse_indiv_ln_bc <- mean(tau_ln_bc^2)
  rmse_indiv_l1_bc <- mean(tau_l1_bc^2)
  rmse_indiv_l2_bc <- mean(tau_l2_bc^2)
  
  # RMSE aggregate
  rmse_aggreg_ln_bc <- mean(tau_ln_bc)^2
  rmse_aggreg_l1_bc <- mean(tau_l1_bc)^2
  rmse_aggreg_l2_bc <- mean(tau_l2_bc)^2
  
  # Bias
  bias_ln_bc <- abs(mean(tau_ln_bc))
  bias_l1_bc <- abs(mean(tau_l1_bc))
  bias_l2_bc <- abs(mean(tau_l2_bc))
  
  rmse_indiv[, t] <- c(rmse_indiv_ln, rmse_indiv_l1, rmse_indiv_l2, rmse_indiv_ln_bc, rmse_indiv_l1_bc, rmse_indiv_l2_bc)
  rmse_aggreg[, t] <- c(rmse_aggreg_ln, rmse_aggreg_l1, rmse_aggreg_l2, rmse_aggreg_ln_bc, rmse_aggreg_l1_bc, rmse_aggreg_l2_bc)
  bias[, t] <- c(bias_ln, bias_l1, bias_l2, bias_ln_bc, bias_l1_bc, bias_l2_bc)
  dens[, t] <- c(dens_ln, dens_l1, dens_l2, NA, NA, NA)
  magnitude[, t] <- c(mag_ln, mag_l1, mag_l2, NA, NA, NA)
  if ((t %% 10) == 0) {
    print(t)
  }
}

averages <- matrix(NA, nrow = 6, ncol = 5)
averages[, 1] <- sqrt(rowMeans(rmse_indiv))
averages[, 2] <- sqrt(rowMeans(rmse_aggreg))
averages[, 3] <- rowMeans(bias)
averages[, 4] <- rowSums(dens)
averages[, 5] <- rowMeans(magnitude)

# Define row and column names
norms <- c('L-infinity', 'L1', 'L2', 'L-infinity-bc', 'L1-bc', 'L2-bc')
metrics <- c('RMSE-indiv', 'RMSE-aggreg', '|Bias|', 'Density', 'Magnitude')
rownames(averages) <- norms
colnames(averages) <- metrics
print(averages)

# End timer and print execution time
end_time <- Sys.time()
execution_time <- end_time - start_time
print(paste("Execution Time: ", execution_time))
