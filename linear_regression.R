# Alex Ramirez
# Final Project
# Linear and Cubic Spline Regression
# CS4342

# imports
library(e1071) # for easy CV

# change the path here, oh benevolent grader
wine <-
  read.csv("skool/temp hw/CS4342/project/winequality-complete.csv")

# set RNG seed
set.seed(69)

# - - - - - - - - - - - - - - - - - Preparation - - - - - - - - - - - - - - - - 
# 5-fold CV definition
five_cv_control <- tune.control(sampling = "cross", cross = 5)

# pair plot (very slow)
# pairs(x=wine, main="Pair plot of Wine Quality")

# print correlations
wine_correlations <- cor(wine)
print(wine_correlations)

# - - - - - - - - - - - - - - - Linear Regression - - - - - - - - - - - - - - -
# fit a CV model: top 5 predictors and all predictors
top5_line_cv <-
  tune(
    METHOD = glm,
    train.x = quality ~ color + volatile.acidity + chlorides + density + alcohol,
    data = wine,
    tunecontrol = five_cv_control
  )
complete_line_cv <-
  tune(
    METHOD = glm,
    train.x = quality ~ .,
    data = wine,
    tunecontrol = five_cv_control
  )

# print estimated MSE
print("top 5 linear MSE")
print(top5_line_cv$best.performance)
print("complete linear MSE")
print(complete_line_cv$best.performance)

# store the best model found
top5_line_model <- top5_line_cv$best.model
complete_line_model <- complete_line_cv$best.model

# make some predictions
top5_line_predictions <-
  predict(object = top5_line_model, x = wine[-13], y = wine[13])
complete_line_predictions <-
  predict(object = complete_line_model, x = wine[-13], y = wine[13])

# evaluate it
plot(
  top5_line_predictions,
  main = "Linear Predictions by Index (Top 5 Predictors)",
  xlab = "Index",
  ylab = "Top 5 Linear Quality Prediction"
)
plot(
  complete_line_predictions,
  main = "Linear Predictions by Index (All Predictors)",
  xlab = "Index",
  ylab = "Complete Linear Quality Prediction"
)

