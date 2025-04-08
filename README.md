# AdamReg: A package for logistic regression with adam stochastic optimization # 

**Installation**

Install the latest version of AdamReg using devtools::install_github("rtucker8/AdamRegression"). You will need to install the devtools package if you have not already.

**Features**

Performs Adam optimization to estimate coefficients for logistic regression models. Optionality to perform Lasso or Ridge regularized regression. For models with a penalty, the cross validation function returns the value of lambda with the highest k-fold cross validated accuracy. Also contains functions for the gradient descent method for performance comparisons.

**Example code**

Simulations_AdamReg.R and heart_AdamReg.R are simulated and real world data examples of the algorithm.

**Acknowledgements**

Authors: Lillian Rountree and Rachel Gonzalez

Made for BIOSTAT 815 in Winter 2025, taught by Michele Peruzzi.

