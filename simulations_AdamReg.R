library(AdamR)
library(tidyverse)
library(microbenchmark)
library(cowplot)
library(glmnet)

#TO DO: Make sure GLM realizes X already has a column for the intercept term

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
  theta.fit <- adam::adam(X, Y, penalty = "none", batch_size = dim(X)[1]/5, tol = 1e-3, alpha=sd(X)/p, maxit = 1000, check_conv = FALSE)
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



# High Dimensional Simulation --------------------------------------------
#Simulate some data
set.seed(2025)

n=1000 #number of rows
p = 99 #number of predictors
k = 10 #number of non-zero coefficients

beta_k_neg = runif(k/2, -2, -.1) #true coefficients, negative
beta_k_pos = runif(k/2, .1, 2) #true coefficients, positive
beta_k = c(beta_k_neg, beta_k_pos) #true coefficients

beta = c(beta_k, rep(0, p-k)) #true coefficients
beta = sample(beta, length(beta))#randomly permute the coefficients
beta = c(runif(1, 0, 1), beta) #add intercept term

rmse_results <- list(glm = numeric(500),  adam = numeric(500), gd = numeric(500))
results = list()

for (i in 1:500) {


  # #Estimation with glm (uses Fisher Scoring to optimize)
  # glm.fit <- cv.glmnet(X[,-1], Y, alpha = 1, family = binomial)
  # results$glm = cbind(results$glm, as.matrix(coef(glm.fit)))
  # #Calculate RMSE
  # fitted.values.glm <- predict(glm.fit, s = "lambda.min", type = "response", newx = X[,-1])
  #
  # rmse_results$glm[i] <- compute_rmse(prob, fitted.values.glm)

  #Estimation with gradient descent
  gradient.opt.lambda <- cross_validate_gd(k=5, X, Y, alpha=0.001, penalty = "lasso", tol=1e-3)
  gradient.fit <- as.matrix(logistic_regression_gd(X, Y, alpha = 0.001, penalty = "lasso", lambda = gradient.opt.lambda, check_conv=FALSE))
  results$gradient = cbind(results$gradient, gradient.fit)
  #Calculate RMSE
  fitted.values.grad <- 1/ (1 + exp(-X %*% gradient.fit))
  rmse_results$gradient[i] <- compute_rmse(prob, fitted.values.grad)

  # #Estimate parameters using our Adam algorithm
  # lambda.opt = adam::cross_validate_adam(k = 5, X, Y, penalty = "lasso", batch_size = dim(X)[1]/5, tol = 1e-3, alpha = sd(X)/p, maxit = 1000)
  # theta.fit <- adam::adam(X, Y, penalty = "lasso", lambda = ,batch_size = dim(X)[1]/5, tol = 1e-3, alpha=sd(X)/p, maxit = 1000, check_conv = FALSE)
  # results$adam = cbind(results$adam, theta.fit)
  # #Calculate RMSE
  # fitted.values.adam <- 1/ (1 + exp(-X %*%theta.fit))
  # rmse_results$adam[i] <- compute_rmse(prob, fitted.values.adam)
  #
}

# Create a data frame for plotting RMSE results
rmse_plot <- data.frame(
  RMSE = c(rmse_results$glm, rmse_results$adam),
  Algorithm = rep(c("glmnet", "AdamReg"), each = 500)
) %>%
  mutate(Algorithm = factor(Algorithm, levels = c("glmnet", "AdamReg"), ordered=T))

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
  ylim(0, .3) +
  guides(fill=FALSE)
print(RMSE)
#Transform results into a single dataframe to plot bias results
glm <- as.data.frame(t(as.matrix(results$glm)))[, which(beta !=0)]
glm_bias <- sweep(glm, 2, beta[which(beta!=0)], "-")
names(glm_bias) <- round(beta[which(beta!=0)], 3)
glm_bias$algorithm = "glmnet"
glm_bias <- glm_bias %>% pivot_longer(cols = -algorithm, names_to = "Variable", values_to = "Bias")

adam <- as.data.frame(t(as.matrix(results$adam)))[, which(beta !=0)]
adam_bias <- sweep(adam, 2, beta[which(beta!=0)], "-")
names(adam_bias) <- round(beta[which(beta!=0)], 3)
adam_bias$algorithm = "AdamReg"
adam_bias <- adam_bias %>% pivot_longer(cols = -algorithm, names_to = "Variable", values_to = "Bias")

bias <- bind_rows(glm_bias, adam_bias)

#Plot Results
bias <- ggplot(bias, aes(x = Variable, y=Bias, fill=algorithm)) +
  geom_boxplot() +
  geom_hline(yintercept = 0, linetype="dashed", color = "black", linewidth = 1.5) +
  scale_fill_brewer(palette = "Set2") +
  ggtitle("Bias of Non-Zero Parameter Estiamtes") +
  xlab("Non-zero Parameter Values") +
  guides(fill=guide_legend(title="Optimization Algorithm")) +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))
print(bias)
# Assess proportion of times the algorithm selected the non-zero values
glm_prop <- as.data.frame(t(as.matrix(results$glm)))[, which(beta !=0)] %>%
  summarise(across(everything(), ~ mean(.x != 0)))
names(glm_prop) <- round(beta[which(beta!=0)], 3)
glm_prop <- glm_prop %>% pivot_longer(cols = everything(), names_to = "Variable", values_to = "Proportion") %>%
  mutate(Algorithm = "glmnet")

#Plot proportion of times the algorithm selected non-zero values as a bar chart
ggplot(glm_prop, aes(x = Variable, y=Proportion, fill=Algorithm)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set2") +
  ggtitle("Selection Probability for Non-Zero Parameters") +
  xlab("Non-zero Parameter Values") +
  guides(fill=guide_legend(title="Optimization Algorithm")) +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12))

#Assess proportion of times the algorithm incorrectly selected zero values
glm_prop <- as.data.frame(t(as.matrix(results$glm)))[, which(beta ==0)] %>%
  summarise(across(everything(), ~ mean(.x != 0)))
glm_prop <- glm_prop %>% pivot_longer(cols = everything(), names_to = "Variable", values_to = "Proportion") %>%
  mutate(Algorithm = "glmnet")


#Plot the proportion of times the algorithm incorrectly selected zero values
ggplot(glm_prop, aes(x = Variable, y=Proportion, fill=Algorithm)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Set2") +
  ggtitle("Selection Probability for Zero Parameters") +
  xlab("Zero Parameter Values") +
  guides(fill=guide_legend(title="Optimization Algorithm")) +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 12),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
