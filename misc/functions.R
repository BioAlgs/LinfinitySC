library(CVXR)
library(doParallel)

solve_column <- function(X0, X_i, n0, lam, norm) {
  # Define the variable W_i (n0 by 1 vector)
  W_i <- Variable(n0)
  # Define the objective function based on the norm type
  if (norm == 'L-infinity') {
    objective <- Minimize(sum_squares(X_i - t(W_i) %*% X0)^2 + lam * p_norm(W_i, "inf"))
  } else if (norm == 'L1') {
    objective <- Minimize(sum_squares(X_i - t(W_i) %*% X0)^2 + lam * sum(W_i))
  } else if (norm == 'L2') {
    objective <- Minimize(sum_squares(X_i - t(W_i) %*% X0)^2 + lam * sum_squares(W_i))
  }
  
  # Define the constraints
  constraints <- list(sum(W_i) == 1, W_i >= 0)
  
  # Define and solve the problem
  problem <- Problem(objective, constraints)
  result <- solve(problem)
  # print(W_i)
  # Return the resulting column as a numeric vector
  return(result$getValue(W_i))
}

solve_w <- function(X1, X0, lam, norm='L-infinity') {
  if (is.data.frame(X0)) {
    X0 <- as.matrix(X0)
  }
  if (is.data.frame(X1)) {
    X1 <- as.matrix(X1)
  }
  n0 <- nrow(X0)
  n1 <- nrow(X1)
  W <- matrix(0, nrow = n0, ncol = n1)
  for (i in 1:n1) {
    X_i <- X1[i, , drop = FALSE]
    W[, i] <- solve_column(X0, X_i, n0, lam, norm)
  }
  return(W)
}

library(doParallel)

solve_w_parallel <- function(X1, X0, lam, norm='L-infinity') {
  if (is.data.frame(X0)) {
    X0 <- as.matrix(X0)
  }
  if (is.data.frame(X1)) {
    X1 <- as.matrix(X1)
  }
  n0 <- nrow(X0)
  n1 <- nrow(X1)
  W <- matrix(0, nrow = n0, ncol = n1)
  
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  
  results <- foreach(i = 1:n1, .packages = c("CVXR"), .export = c("solve_column")) %dopar% {
    X_i <- X1[i, , drop = FALSE]
    solve_column(X0, X_i, n0, lam, norm)
  }
  
  stopCluster(cl)
  
  for (i in 1:n1) {
    W[, i] <- results[[i]]
  }
  
  return(W)
}
