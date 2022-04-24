library(tidyverse)
library(glmnet)
library(caret)

data <- read.csv("fbbdata.csv")
glimpse(data)
data <- select(data, -c("val_per_sp", "val_per_of", "val_per_c"))
glimpse(data)

data <- na.omit(data) # omit na values
data$total_points <- log(data$total_points) # As the dependent variable
# has very large values, consider logarithmic transformation to reduce
# RMSE.

# Split the data into train and test sets
## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]

# Train a lasso regression model

d <- select(train,-c("total_points"))
x_train <- as.matrix(d)
y_train <- (train$total_points)

d <- select(test,-c("total_points"))
x_test <- as.matrix(d)
y_test <- (test$total_points)

lambdas <- 10^seq(2, -3, by = -.1)

# Fit a lasso regression model
m1 <- glmnet(x_train, y_train, standardize = TRUE, nfolds = 5, alpha = 1, lambda = lambdas)

# Get optimized lambda value by cv method
cv <- cv.glmnet(x_train, y_train, alpha = 1, lambda = lambdas)
optimized <- cv$lambda.min
optimized

# Get prediction accuracy on test set
pred <- m1 %>% predict(x_test, s = optimized)
rmse <- RMSE(pred, test$total_points)
mae <- MAE(pred, test$total_points)
data.frame(Measure = c("RMSE", "MAE"),
           Result = c(rmse, mae)
)

# Ridge regression
m2 <- glmnet(x_train, y_train,alpha = 0, nlambda = 25, lambda = lambdas)

cv2 <- cv.glmnet(x_train, y_train, alpha = 0, nlambda = 25, lambda = lambdas)
optimized2 <- cv2$lambda.min
optimized2

# Get prediction accuracy on test set
pred2 <- m2 %>% predict(x_test, s = optimized2)
rmse2 <- RMSE(pred2, test$total_points)
mae2 <- MAE(pred2, test$total_points)
data.frame(Measure = c("RMSE", "MAE"),
           Result = c(rmse2, mae2)
)

# Elastic net regression
set.seed(123)
m3 <- train(
  total_points ~., data = train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Best tuning parameter
m3$bestTune
pred3 <- m3 %>% predict(x_test, s = 0.02166497)
rmse3 <- RMSE(pred3, test$total_points)
mae3 <- MAE(pred3, test$total_points)
data.frame(Measure = c("RMSE", "MAE"),
           Result = c(rmse3, mae3)
)



# Evaluating the RMSE and MAE of these three regression models, 
# Ridge regression has shown optimal result. The best parameter was tuned
# using CV method for each model.

