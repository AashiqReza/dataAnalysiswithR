library(tidyverse)
library(caret)
library(e1071)
library(caret)
library(randomForest)
library(rnn)
library(rpart)
library(validann)
library(neuralnet)
library(keras)
library(FSelectorRcpp)
library(nnet)
library(party)


d <- read.csv("heart_disease_health_indicators_BRFSS2015.csv")

glimpse(d)
d$HeartDiseaseorAttack <- ifelse(d$HeartDiseaseorAttack == 0, "No", "Yes")
d$HighBP <- as.factor(d$HighBP)
d$HighChol <- as.factor(d$HighChol)
d$Smoker <- as.factor(d$Smoker)
d$Stroke <- as.factor(d$Stroke)
d$Diabetes <- as.factor(d$Diabetes)
d$HvyAlcoholConsump <- as.factor(d$HvyAlcoholConsump)


## Plotting the heart attack ratio

atk <- d %>% 
  group_by(HeartDiseaseorAttack) %>% 
  summarise(Count = n())%>% 
  mutate(percentage = prop.table(Count)*100)
g1 <- ggplot(atk, aes(reorder(HeartDiseaseorAttack, -percentage), percentage))+
  geom_col(col = c("seagreen", "maroon"), fill = c("seagreen","maroon"), width = 0.5)+
  geom_text(aes(label = sprintf("%.2f%%", percentage)), color = "white", size = 6, 
            position=position_dodge(width = 0.9), 
            vjust=1.3)+
  scale_colour_manual(values=c("#FF5733", "#000000"))+
  xlab("Diseased or not") + 
  ylab("Percent")+
  ggtitle("Heart Diseased Percentage") +
  theme_bw()
g1

#ggsave("percentage.bmp", dpi = 1000, height = 3, width = 3.3)


## 
d$Sex <- as.factor(d$Sex)
d$Sex <- ifelse(d$Sex == 0, "Female", "Male")


g2 <- ggplot(d, aes(x=Sex, fill=HeartDiseaseorAttack))+
         geom_bar(position = "dodge") +
  theme_minimal() + scale_fill_manual(values = c("steelblue","brown")) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10), 
        legend.title = element_text( size=0)) 
g2

##
d$HighBP <- as.factor(d$HighBP)
d$HighBP <- ifelse(d$HighBP == 0, "No", "Yes")

g3 <- ggplot(d, aes(x=HighBP,fill=HeartDiseaseorAttack))+
  geom_bar(position = "dodge") +
  theme_minimal() + scale_fill_manual(values = c("steelblue","brown")) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10), 
        legend.title = element_text( size=0)) 
g3

##
d$HighChol <- ifelse(d$HighChol == 0, "No", "Yes")


g4 <- ggplot(d, aes(x=HighChol,fill=HeartDiseaseorAttack))+
  geom_bar(position = "dodge") +
  theme_minimal() + scale_fill_manual(values = c("steelblue","brown")) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10), 
        legend.title = element_text( size=0)) 
g4

##

d$Smoker <- ifelse(d$Smoker == 0, "No", "Yes")


g5 <- ggplot(d, aes(x=Smoker,fill=HeartDiseaseorAttack))+
  geom_bar(position = "dodge") +
  theme_minimal() + scale_fill_manual(values = c("steelblue","brown")) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10), 
        legend.title = element_text( size=0))
g5


##

d$Stroke <- ifelse(d$Stroke == 0, "No", "Yes")


g6 <- ggplot(d, aes(x=Stroke,fill=HeartDiseaseorAttack))+
  geom_bar(position = "dodge") +
  theme_minimal() + scale_fill_manual(values = c("steelblue","brown")) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10), 
        legend.title = element_text( size=0))
g6


##

d$Diabetes <- ifelse(d$Diabetes == 0, "No", "Yes")


g7 <- ggplot(d, aes(x=Diabetes,fill=HeartDiseaseorAttack))+
  geom_bar(position = "dodge") +
  theme_minimal() + scale_fill_manual(values = c("steelblue","brown")) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10), 
        legend.title = element_text( size=0))
g7


##

d$PhysActivity <- ifelse(d$PhysActivity == 0, "No", "Yes")


g8 <- ggplot(d, aes(x=HeartDiseaseorAttack,y = BMI,
                    fill=HeartDiseaseorAttack))+
  geom_boxplot() +
  theme_minimal() + scale_fill_manual(values = c("steelblue","brown")) +
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10), 
        legend.title = element_text(size=0))

g8

library(ggpubr)
fig1 <- ggarrange(g2, g3, g4, g5, g6, g7, common.legend = TRUE, legend="bottom")
annotate_figure(fig1, top  = text_grob("Varying rate of diseased in different groups of people",
                                       col = "black", face = "bold", size = 10))
## ggsave("onfact.bmp", dpi = 1000, height = 3.5, width = 3.4)

##### Classification models

# TRAIN TEST
d <- read.csv("heart_disease_health_indicators_BRFSS2015.csv")
d$HeartDiseaseorAttack <- as.factor(d$HeartDiseaseorAttack)
d$HeartDiseaseorAttack <- ifelse(d$HeartDiseaseorAttack == 0, "No", "Yes")
d$HighBP <- as.factor(d$HighBP)
d$HighChol <- as.factor(d$HighChol)
d$Smoker <- as.factor(d$Smoker)
d$Stroke <- as.factor(d$Stroke)
d$Diabetes <- as.factor(d$Diabetes)
d$HvyAlcoholConsump <- as.factor(d$HvyAlcoholConsump)
d$PhysActivity <- as.factor(d$PhysActivity)
d$Fruits <- as.factor(d$Fruits)
d$Veggies <- as.factor(d$Veggies)
d$AnyHealthcare <- as.factor(d$AnyHealthcare)
d$NoDocbcCost <- as.factor(d$NoDocbcCost)
d$GenHlth <- as.factor(d$GenHlth)
d$DiffWalk <- as.factor(d$DiffWalk)
d$Sex <- as.factor(d$Sex)
d$Education <- as.factor(d$Education)
d$Income <- as.factor(d$Income)

d <- d[1:10000,]
## 70% of the sample size
smp_size <- floor(0.30 * nrow(d))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = smp_size)

train <- d[train_ind, ]
test <- d[-train_ind, ]



## TREE
m1 <- rpart(HeartDiseaseorAttack ~.,train, 
            control = rpart.control(minsplit = 1, minbucket = round(1)/3,
                                    cp = 0.000001, maxdepth = 30,
                                    xval = 20))
p1 <- predict(m1, test, type = "class")
cm <- table(p1, test$HeartDiseaseorAttack)
cm
confusionMatrix(cm)

## FORREST
m2 <- randomForest(as.factor(HeartDiseaseorAttack)~., train)
p <- predict(m2, test, type = "prob")
pp <- p[,2] >= 0.22
p2 <- as.numeric(pp)
p2 <- ifelse(p2 == FALSE, "No", "Yes")

cm <- table(p2, test$HeartDiseaseorAttack)
cm
confusionMatrix(cm)


## SVM
train$HeartDiseaseorAttack <- as.factor(train$HeartDiseaseorAttack)
m4 <- svm(HeartDiseaseorAttack~., train, type = "C-classification",
          kernel = "sigmoid", coef0 = 0.1)
p4 <- predict(m4, test, type = "terms", interval = "prediction",level = 0.85)
cm <- table(p4, test$HeartDiseaseorAttack)
cm
confusionMatrix(cm)


## glm
train$HeartDiseaseorAttack <- as.factor(train$HeartDiseaseorAttack)
g <- glm(HeartDiseaseorAttack~., train, family = "binomial")
pg <- predict(g, test, type = "response")
pgx <- pg >= 0.15
pgp <- as.numeric(pgx)
pgp <- ifelse(pgp == FALSE, "No", "Yes")
cm <- table(pgp, test$HeartDiseaseorAttack)
confusionMatrix(cm)


## XGB
library(xgboost)
d <- read.csv("heart_disease_health_indicators_BRFSS2015.csv")
smp_size <- floor(0.70 * nrow(d))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = smp_size)

train <- d[train_ind, ]
test <- d[-train_ind, ]


y_train <- as.integer(train$HeartDiseaseorAttack) 
y_test <- as.integer(as.factor(test$HeartDiseaseorAttack))
X_train <- train %>% select(-HeartDiseaseorAttack)
X_test <- test %>% select(-HeartDiseaseorAttack)

xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)
xgb_params <- list(
  booster = "gbtree",
  eta = 0.01,
  max_depth = 8,
  gamma = 4,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 2
)
xgb_model <- xgb.train(
  params = xgb_params,
  data = xgb_train,
  nrounds = 5000,
  verbose = 1
)

xgb_preds <- predict(xgb_model, as.matrix(X_test), reshape = TRUE)
xgb_preds <- as.data.frame(xgb_preds)
px <- xgb_preds[,2] >= 0.15
p5 <- as.numeric(px)
cm <- table(p5, test$HeartDiseaseorAttack)
confusionMatrix(cm)


## ANN
library(neuralnet)


d <- read.csv("heart_disease_health_indicators_BRFSS2015.csv")
d <- d[1:10000,]
smp_size <- floor(0.70* nrow(d))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(d)), size = smp_size)

train <- d[train_ind, ]
test <- d[-train_ind, ]

set.seed(123)
nn <- neuralnet((HeartDiseaseorAttack) ~., 
                     data = train, 
                     linear.output = FALSE, 
                     err.fct = 'sse',
                     act.fct = "tanh",
                     likelihood = TRUE)

np <- compute(nn, test[,-1])
pnp <- np$net.result < 0
pnn <- as.numeric(pnp)
cm <- table(pnn, test$HeartDiseaseorAttack)
cm
confusionMatrix(cm)

train_params <- trainControl(method = "repeatedcv", number = 10, repeats=5)

my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))
nnet_model <- train(train[,-1], train$HeartDiseaseorAttack,
                    method = "nnet",
                    trControl= train_params,
                    preProcess=c("pca"),
                    na.action = na.omit,
                    maxit = 1000, trace = F, linout = 1
)
nnet_predictions_train <-predict(nnet_model, train)
head(nnet_predictions_train)

pnp <- nnet_predictions_train > 0.175
pnn <- as.numeric(pnp)
cm <- table(pnn, train$HeartDiseaseorAttack)
cm
confusionMatrix(cm)




