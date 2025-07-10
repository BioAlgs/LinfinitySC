library("Synth")
library("kernlab")

# X0 <- as.matrix(read.csv('../Tobacco/X0.csv'))
# X1 <- as.matrix(read.csv('../Tobacco/X1.csv'))
# Z0 <- as.matrix(read.csv('../Tobacco/Z0.csv'))
# Z1 <- as.matrix(read.csv('../Tobacco/Z1.csv'))
# Y0 <- as.matrix(read.csv('../Tobacco/Y0.csv'))
# Y1 <- as.matrix(read.csv('../Tobacco/Y1.csv'))

X0 <- as.matrix(read.csv('..\\Tobacco\\X0.csv'))
X1 <- as.matrix(read.csv('..\\Tobacco\\X1.csv'))
Z0 <- as.matrix(read.csv('..\\Tobacco\\Z0.csv'))
Z1 <- as.matrix(read.csv('..\\Tobacco\\Z1.csv'))
Y0 <- as.matrix(read.csv('..\\Tobacco\\Y0.csv'))
Y1 <- as.matrix(read.csv('..\\Tobacco\\Y1.csv'))
options(scipen = 999)
source("synth.R")
res <- synth(X1 = X1, X0 = X0, 
             Z0 = Z0, Z1 = Z1, 
             custom.v = rep(1, nrow(X0)) / nrow(X0), 
             optimxmethod = "BFGS", 
             genoud = FALSE, quadopt = "ipop", 
             Margin.ipop = 5e-04, 
             Sigf.ipop = 5, 
             Bound.ipop = 10)

res$loss.w
as.matrix(res$solution.v)
as.matrix(res$solution.w)
