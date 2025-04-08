# Functions for AdamRegression package

#' Calculate gradient of the loss function for logistic regression.
#'
#' @description Calculates the gradient of the objective function for logistic regression.
#'
#' @details The gradient without is: \deqn{-X^T(Y-\mu)}
#'
#'
#' where \eqn{\mu = \frac{e^{X^T\beta}}{1+e^{X^T\beta}}}.
#'
#' @param theta A parameter vector.
#' @param X A matrix of features.
#' @param Y The response vector.
#' @returns A numeric value.
#' @export
gradient_function <- function(theta, X, Y) {

  eta <- X %*% theta # Logistic function
  mu <- 1 / (1 + exp(-eta))
  grad <- t(X) %*% (mu - Y)
  return(grad)
}

#' Perform gradient descent optimization for logistic regression.
#'
#' @description Implements the gradient descent algorithm for logistic regression with a fixed learning rate.
#'
#' @details The gradient without penalties is: \deqn{-X^T(Y-\mu)}
#'
#' where \eqn{\mu = \frac{e^{X^T\beta}}{1+e^{X^T\beta}}}.
#'
#' The update step is given by \deqn{\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)}
#'
#' @param X A matrix of features.
#' @param Y The response vector, binary.
#' @param alpha The learning rate (step size), fixed.
#' @param maxit The maximum number of iterations.
#' @param tol The tolerance for convergence.
#' @param check_conv When TRUE, reports the number of iterations needed for convergence.
#' @returns A numeric value.
#' @export
logistic_regression_gd <- function(X, Y, alpha = 0.01, maxit = 1000, tol = 1e-6, check_conv = TRUE) {

  # Initialize parameters (starting with zeros)
  theta <- rnorm(ncol(X), 0, 0.01)

  # Initialize some variables to track progress
  prev_theta <- theta
  converged <- FALSE

  for (iter in 1:maxit) {
    # Compute the gradient
    grad <- gradient_function(theta, X, Y)

    # Update parameters using gradient descent
    theta <- theta - alpha * grad

    # Check for convergence (if change in parameters is small)
    if (max(abs(grad)) < tol) {
      converged <- TRUE
      break
    }

    # Update prev_theta for the next iteration
    prev_theta <- theta
  }

  # Return the fitted parameters (theta)
  if (converged) {
    if (check_conv) {
      cat("Gradient Descent Converged in", iter, "iterations.\n")}
  } else {
    cat("Gradient Descent did not converge within the maximum number of iterations.\n")
  }
  return(theta)
}


#' Adaptive moment estimation for optimization of logistic regression
#'
#' @description Estimates the coefficients of logistic regression, using an algorithm
#' for first-order gradient-based optimization of stochastic objective functions,
#' based on adaptive estimates of lower-order moments.
#' From Kingma, D. P., & Ba, J. (2014).
#' Adam: A Method for Stochastic Optimization.
#' https://arxiv.org/abs/1412.6980.
#'
#' @details Default values for parameters based on suggestions from Kingma and Ba (2014).
#'
#' @param X A matrix of features.
#' @param Y The response vector.
#' @param batch_size Size of mini-batches used in the gradient descent algorithm.
#' @param alpha The learning rate (step size).
#' @param beta_decay The exponential decay rates for moment estimates. Must be in [0, 1).
#' @param epsilon Small value to avoid division by zero. Must be positive.
#' @param maxit Maximum number of iterations.
#' @param tol Tolerance for convergence.
#' @param check_conv When TRUE, reports the number of iterations needed for convergence.
#' @returns A numeric vector of estimated coefficients.
#' @export
adam <- function(X, Y, batch_size = 32, alpha = 0.001, beta_decay = c(0.9, 0.999),
                 epsilon = 1e-8, maxit = 1000, tol= 1e-4, check_conv = TRUE) {

  # Check that elements of beta are in  [0,1)
  if (any(beta_decay >= 1) || any(beta_decay < 0)) {
    stop("Elements of beta must be in [0,1)")
  }

  # Check that epsilon is positive
  if (epsilon <= 0) {
    stop("epsilon must be positive")
  }

  #Check that X and Y have the same number of rows
  if (nrow(X) != length(Y)) {
    stop("X and Y must have the same number of rows")
  }

  #If X does not include column of ones for intercept, add it now
  if (all(X[,1] != 1)) {
    X <- cbind(1, X)
    print("Adding column of ones to design matrix.")
  }

  #check that batch size is between 1 and the number of observations
  if (batch_size < 1 || batch_size > nrow(X)) {
    stop("Batch size must be between 1 and the number of observations in X.")
  }

  # Characteristics of data
  p <- ncol(X)
  n <- nrow(X)

  # Initialize parameter vector
  # theta <- rep(0, ncol(X))
  theta <- runif(ncol(X), min=-1, max=1)

  #Initialize 1st and 2nd moment vectors
  m <- rep(0, length(theta))
  v <- rep(0, length(theta))

  #Initialize time step
  t <- 0

  #While theta_t not converged and t < maxit do
  while (t < maxit) {

    t = t + 1 #Increment time step

    #Take random subsample (mini batches) of size batch_size
    ids <- sample(1:n, batch_size)
    X_sub <- X[ids,]
    Y_sub <- Y[ids]

    #Compute gradient
    #grad <- logistic_grad(theta, X_sub, Y_sub)
    grad <- gradient_function(theta, X_sub, Y_sub)

    #Update biased first and second raw moment estimates
    m <- beta_decay[1] * m + (1 - beta_decay[1]) * grad
    v <- beta_decay[2] * v + (1 - beta_decay[2]) * grad^2

    #bias corrected moment estimates
    m_hat <- m/(1-beta_decay[1]^t)
    v_hat <- v/(1-beta_decay[2]^t)

    #Update Parameter Vector
    theta_new <- theta - alpha * (1/sqrt(t)) * (m_hat / (sqrt(v_hat) + epsilon))

    #Assess convergence
    if (sum(apply(theta_new - theta, 2, function(x) x < tol)) == dim(X)[2]){
      if (check_conv == TRUE){
      print(paste("Converged in", t, "iterations"))}
      return(theta_new)
      break
    }

    theta <- theta_new
  }

  print(paste0("Algorithm did not converge in ", maxit, " iterations."))
  return(theta)
}
