library(AdamReg)
library(tidyverse)

## Heart attack data from Indonesia
## Source: https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-indonesia

# Read in data
df <- read.csv("heart.csv")

# Add intercept term
df <- as.data.frame(cbind(intercept=rep(1, dim(df)[1]), df))

# Binarize gender
df$gender <- as.numeric(df$gender=="Female")

# Scale continuous covariates for model
df$age <- scale(df$age)
df$blood_pressure_systolic <- scale(df$blood_pressure_systolic)


# Prepare data for adam
y <- df$heart_attack
x <- as.matrix(df %>% select(intercept, age, blood_pressure_systolic, gender, diabetes, family_history, 
                             obesity, previous_heart_disease))
p <- 7

# Adam
theta.fit <- adam(x, y, penalty = "none", batch_size = dim(x)[1]/5, tol = 10e-5, 
                  alpha=min(sd(x)/p, 1),
                  maxit = 5000, check_conv=TRUE)

# GLM
glm.fit <- glm(heart_attack ~ age + blood_pressure_systolic + gender + diabetes + family_history + 
                 obesity + previous_heart_disease, data=df, family = binomial(link = "logit")) 

# Compare
data.frame(glm=round(coef(glm.fit),2), adam=round(theta.fit, 2))
