library(CVXR)
# Set the number of observations and predictors
n <- 100  # Number of observations
p <- 5    # Number of predictors (make sure p <= n)

# Generate a random matrix
set.seed(123)
X <- matrix(rnorm(n * p), nrow = n, ncol = p)

# Use QR decomposition to get an orthonormal matrix
qrX <- qr(X)
Q <- qr.Q(qrX)
X_orthonormal <- Q[, 1:p]  # Ensure X is orthonormal

# Check orthonormality
t(X_orthonormal) %*% X_orthonormal  # Should be close to the identity matrix

# Generate coefficients
beta <- c(-1, -2, 0, 0, 1)
# Generate the response variable y
y <- X_orthonormal %*% beta # + rnorm(n)  # Adding noise

# CVXR variables
beta <- Variable(p)
lambda <-  2.6 # 2.0 # 2.6 

# Objective: Minimize the sum of squared errors
objective <- Minimize(sum((y - X_orthonormal %*% beta)^2) + lambda * cvxr_norm(beta, "inf"))

# Formulate the problem
problem <- Problem(objective)

# Solve the problem
result <- solve(problem)
# Display results
# cat("The optimal coefficients beta are:\n")
print(result$getValue(beta))# CVXR variables

# beta.hat <- t(X_orthonormal) %*% y
