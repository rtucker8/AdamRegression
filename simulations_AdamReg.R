library(AdamR)
library(tidyverse)
library(microbenchmark)
library(cowplot)
library(glmnet)

#Simulate some data
set.seed(2025)

n=600 #number of rows
p = 5 #number of predictors
beta = c(.16, .42, -1.52, 1.08, .09, -.85) #true coefficients for centered and scaled covariates

#Negative log-likelihood function for our logistic regression problem, for use with AdamR function
nll <- function(theta, data) {
  theta <- as.numeric(theta)
  Y = data[,1]
  Y = as.matrix(Y)
  X = data[,-1]
  X = as.matrix(X)
  linear_pred <- X %*% theta
  prob <- 1 / (1 + exp(-linear_pred))  # sigmoid function
  cost <- -sum(Y * log(prob) + (1 - Y) * log(1 - prob))  # Negative log-likelihood
  cost <- as.numeric(cost)
  return(cost)
}

# RMSE calculation function
compute_rmse <- function(true_values, estimated_values) {
  return(sqrt(mean((true_values - estimated_values)^2)))
}

rmse_results <- list(glm = numeric(500), gradient = numeric(500), adamR = numeric(500), adam = numeric(500))
results = list()

for (i in 1:500) {

  print(i)

  X = matrix(rnorm(n*p), n, p)
  X = scale(X)
  X = cbind(1, X)

  prob = 1/(1+exp(-(X%*%beta)))
  Y = rbinom(n, 1, prob)

  myData = cbind(Y, X)
  myData = as.matrix(myData)

  #Estimation with glm (uses Fisher Scoring to optimize)
  glm.fit <- glm(Y ~ X, family = binomial)
  results$glm = cbind(results$glm, as.matrix(summary(glm.fit)$coefficients[,1]))
  #Calculate RMSE
  fitted.values.glm <- glm.fit$fitted.values
  rmse_results$glm[i] <- compute_rmse(prob, fitted.values.glm)

  #Estimation with gradient descent
  gradient.fit <- as.matrix(logistic_regression_gd(X, Y, alpha = 0.001, check_conv=FALSE))
  results$gradient = cbind(results$gradient, gradient.fit)
  #Calculate RMSE
  fitted.values.grad <- 1/ (1 + exp(-X %*% gradient.fit))
  rmse_results$gradient[i] <- compute_rmse(prob, fitted.values.grad)

  #Estimation with AdamR algorithm from GitHub
  AdamR.fit = AdamR::AdamR(theta = rep(0, p+1),
                       f = function(theta) nll(theta, myData),
                       data = myData, batch.size = dim(myData)[1]/5,
                       alpha = sd(myData[,-1])/p,
                       beta1=0.9,
                       beta2 = 0.999,
                       thres = 1e-3,
                       maxepoch = 1000 )
  results$adamR = cbind(results$adamR, AdamR.fit$best.theta)
  #Calculate RMSE
  fitted.values.adamR <- 1/ (1 + exp(-X %*% AdamR.fit$best.theta))
  rmse_results$adamR[i] <- compute_rmse(prob, fitted.values.adamR)

  #Estimate parameters using our Adam algorithm
  theta.fit <- adam::adam(X, Y, batch_size = dim(X)[1]/5, tol = 1e-3, alpha=sd(X)/p, maxit = 1000, check_conv = FALSE)
  results$adam = cbind(results$adam, theta.fit)
  #Calculate RMSE
  fitted.values.adam <- 1/ (1 + exp(-X %*%theta.fit))
  rmse_results$adam[i] <- compute_rmse(prob, fitted.values.adam)

}

# Create a data frame for plotting RMSE results
rmse_plot <- data.frame(
  RMSE = c(rmse_results$glm, rmse_results$gradient, rmse_results$adamR, rmse_results$adam),
  Algorithm = rep(c("glm", "gradient descent", "AdamR", "AdamReg"), each = 500)
) %>%
  mutate(Algorithm = factor(Algorithm, levels = c("glm", "gradient descent", "AdamR", "AdamReg"), ordered=T))

# Plot RMSE for each algorithm
RMSE <- ggplot(rmse_plot, aes(x = Algorithm, y = RMSE, fill = Algorithm)) +
  geom_boxplot() +
  ggtitle("RMSE of Estimates") +
  scale_fill_brewer(palette = "Set2") +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        axis.text.x = element_text(angle = 45),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12)) +
  ylim(0, .1) +
  guides(fill=FALSE)

#Transform results into a single dataframe to plot bias results
glm <- as.data.frame(t(as.matrix(results$glm))) %>% mutate(algorithm = "glm")
names(glm) <- c("V1", "V2", "V3", "V4", "V5", "V6", "algorithm")
gradient <- as.data.frame(t(as.matrix(results$gradient))) %>% mutate(algorithm = "gradient descent")
adamR <- as.data.frame(t(as.matrix(results$adamR))) %>% mutate(algorithm = "AdamR")
adam <- as.data.frame(t(as.matrix(results$adam))) %>% mutate(algorithm = "AdamReg")

bias_plot <- rbind(glm, gradient, adamR, adam) %>% mutate(V1 = as.numeric(V1) - beta[1],
                                                      V2 = as.numeric(V2) - beta[2],
                                                      V3 = as.numeric(V3) - beta[3],
                                                      V4 = as.numeric(V4) - beta[4],
                                                      V5 = as.numeric(V5) - beta[5],
                                                      V6 = as.numeric(V6) - beta[6],
                                                      algorithm = factor(algorithm, levels = c("glm", "gradient descent", "AdamR", "AdamReg"), ordered=T)) %>%
pivot_longer(cols = c(V1, V2, V3, V4, V5, V6), names_to = "Variable", values_to = "Bias")

#Plot Results
bias <- ggplot(bias_plot, aes(x = Variable, y=Bias, fill=algorithm)) +
  geom_boxplot() +
  geom_hline(yintercept = 0, linetype="dashed", color = "black", linewidth = 1.5) +
  scale_fill_brewer(palette = "Set2") +
  ggtitle("Bias of Estimates") +
  guides(fill=guide_legend(title="Optimization Algorithm")) +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12))

plot_grid(RMSE, bias, labels = c("A", "B"), label_size = 16, rel_widths = c(1, 2) )

ggsave("RMSE_Bias_plot.png", width = 12, height = 6)

#Perform benchmarking analysis to compare algorithm speed
set.seed(2025)
n=600 #number of rows
p = 5 #number of predictors
beta = c(.16, .42, -1.52, 1.08, .09, -.85) #true coefficients
X = matrix(rnorm(n*p), n, p)
X = cbind(1, X)
prob = 1/(1+exp(-(X%*%beta)))
Y = rbinom(n, 1, prob)
myData = cbind(Y, X)
myData = as.matrix(myData)

benchmark_results <- microbenchmark(
  AdamR = {
    AdamR.fit = AdamR::AdamR(theta = rep(0, p+1),
                       f = function(theta) nll(theta, myData),
                       data = myData, batch.size = dim(myData)[1]/5,
                       alpha = sd(myData[,-1])/p,
                       beta1=0.9,
                       beta2 = 0.999,
                       thres = 1e-3,
                       maxepoch = 1000)
  },
  adam = {
    theta.fit <- adam::adam(X, Y, penalty = "none", batch_size = dim(X)[1]/5, tol = 1e-3, alpha=sd(X)/p, maxit = 5000, check_conv = FALSE)
  },
  gradient = {
    gradient.fit <- as.matrix(logistic_regression_gd(X, Y, alpha = 0.001, check_conv = FALSE))
  },
  glm = {
    glm.fit <- glm(Y ~ X, family = binomial)
  },
  times=100
)
benchmark_results <- as.data.frame(benchmark_results) %>% mutate(time_microsec = time/1000,
                                                                 algorithm = case_when(
                                                                   expr == "AdamR" ~ "AdamR",
                                                                   expr == "adam" ~ "AdamReg",
                                                                   expr == "gradient" ~ "Gradient Descent",
                                                                   expr == "glm" ~ "glm"
                                                                 ))
ggplot(benchmark_results, aes(x = algorithm, y=time_microsec, fill=algorithm)) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  ggtitle("Efficiency of Algorithms after 100 Repititions") +
  labs(x = "Optimization Algorithm", y = "Time (microseconds)" ) +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12)) +
  guides(fill=FALSE)
ggsave("benchmark_plot.png", width = 9, height = 6)
